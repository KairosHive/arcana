import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import pandas as pd
import cv2
import base64
import os
import pickle
import torch
from usearch.index import Index
from transformers import CLIPModel, CLIPProcessor
from concurrent.futures import ThreadPoolExecutor
import dash_daq as daq
import numpy as np
import plotly.graph_objects as go
import re

torch.set_grad_enabled(False)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LATENTS_DIR = os.path.join(APP_ROOT, "latents")
DB_DIR = os.path.join(APP_ROOT, "databases")

OUTPUT_DIR = os.path.join(APP_ROOT, "output")
STORIES_DIR = os.path.join(OUTPUT_DIR, "stories")
SELECTIONS_DIR = os.path.join(OUTPUT_DIR, "selections")

os.makedirs(STORIES_DIR, exist_ok=True)
os.makedirs(SELECTIONS_DIR, exist_ok=True)


# ------------- FILE DISCOVERY HELPERS -------------
def get_latent_options(latent_dir=LATENTS_DIR, n_dim=2):
    pattern = re.compile(rf"latent_space_(.+)_{n_dim}d\.pkl$")
    options = []
    for fname in os.listdir(latent_dir):
        m = pattern.match(fname)
        if m:
            options.append({"label": m.group(1), "value": m.group(1)})
    return sorted(options, key=lambda x: x["label"])

def get_db_options(db_dir=DB_DIR):
    pattern = re.compile(r"index_(.+)\.pkl$")
    options = []
    for fname in os.listdir(db_dir):
        m = pattern.match(fname)
        if m:
            options.append({"label": m.group(1), "value": m.group(1)})
    return sorted(options, key=lambda x: x["label"])

def get_matching_datasets(latent_dir=LATENTS_DIR, db_dir=DB_DIR):
    latent_pattern = re.compile(r"latent_space_(.+)_(\d+)d\.pkl$")
    latent_map = {}
    for fname in os.listdir(latent_dir):
        m = latent_pattern.match(fname)
        if m:
            name, dim = m.group(1), m.group(2)
            latent_map.setdefault(name, []).append(dim)
    # Now find intersection with databases
    db_pattern = re.compile(r"index_(.+)\.pkl$")
    db_names = {m.group(1) for fname in os.listdir(db_dir) if (m := db_pattern.match(fname))}
    options = []
    for name, dims in latent_map.items():
        if name in db_names:
            for dim in sorted(dims):
                label = f"{name} ({dim}D)"
                value = f"{name}::{dim}"
                options.append({"label": label, "value": value})
    return sorted(options, key=lambda x: x["label"])

dataset_options = get_matching_datasets()
default_dataset = dataset_options[0]['value'] if dataset_options else None

# ------------- DATA LOADING HELPERS -------------
def encode_thumbnail(path, max_side=128):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    thumb = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    x_offset = (max_side - new_w) // 2
    y_offset = (max_side - new_h) // 2
    thumb[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
    _, buffer = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"

def load_data(name, n_dim=2):
    latent_path = os.path.join(LATENTS_DIR, f"latent_space_{name}_{n_dim}d.pkl")
    latent_path_thumb = latent_path.replace('.pkl', '_thumbnail.pkl')
    if os.path.exists(latent_path_thumb):
        df = pd.read_pickle(latent_path_thumb)
    else:
        df = pd.read_pickle(latent_path)
        paths = df['path'].tolist()
        with ThreadPoolExecutor(max_workers=8) as executor:
            thumbnails = list(executor.map(encode_thumbnail, paths))
        df['thumbnail'] = thumbnails
        df.to_pickle(latent_path_thumb)
    df['label'] = df['label'].astype(str)
    return df

def load_index(name):
    index_name = os.path.join(DB_DIR, f"index_{name}.pkl")
    with open(index_name, "rb") as f:
        index, idx2path = pickle.load(f)
    index = Index.restore(index)
    return index, idx2path

def encode_image(image_path):
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return base64.b64encode(buffer).decode()

# ------------- CLIP MODEL LOAD ONCE -------------
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16)
model.eval().to("cuda")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

def search(index, idx2path, query, n):
    inputs = processor.tokenizer(query, return_tensors="pt").to("cuda")
    vec = model.get_text_features(**inputs).detach().cpu().numpy().flatten()
    idxs = index.search(vec, n, exact=True)
    return [(idx.key, idx2path[idx.key], idx.distance) for idx in idxs]

# ------------- DASH APP -------------
latent_options = get_latent_options()
db_options = get_db_options()
default_latent = latent_options[0]['value'] if latent_options else ''
default_db = db_options[0]['value'] if db_options else ''

app = dash.Dash(
    __name__,
    external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
    suppress_callback_exceptions=True,
)

app.layout = html.Div(style={'backgroundColor': '#121212', 'color': 'white', 'padding': '20px', 'height': '100vh'}, children=[
    html.Div([
        dcc.RadioItems(
            id='mode-select',
            options=[
                {'label': 'Prompt Search', 'value': 'prompt'},
                {'label': 'Generate Story', 'value': 'story'}
            ],
            value='prompt',
            labelStyle={'display': 'inline-block', 'marginRight': '25px', 'fontWeight': 'bold'}
        ),
        html.Label("Dataset:", style={'marginLeft': '40px', 'marginRight': '6px'}),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=get_matching_datasets(),
            value=None,   # set default below after checking
            clearable=False,
            style={'width': '220px', 'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#000'}
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),


    dcc.Store(id='story-cache', storage_type='memory'),

    html.Div([
        dcc.Graph(id='scatter-plot', style={'height': '80vh'}),
        html.Div([
            dcc.Input(id='search-box', type='text', placeholder='Enter a prompt...', style={'width': '60%', 'marginRight': '10px'}),
            dcc.Input(id='num-images', type='number', value=4, min=1, max=20, style={'width': '15%', 'marginRight': '10px'}),
            dcc.Textarea(
                id='story-box',
                placeholder='Enter your story, one scene per line. (Press ENTER after each scene.)',
                style={'width': '70%', 'height': '70px', 'marginRight': '10px'}
            ),
            html.Button('Search', id='main-action-btn', n_clicks=0)
        ], id='controls-bar', style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})
    ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.Div(id='image-display', style={'overflowY': 'scroll', 'maxHeight': '80vh'}),
        html.Button('Save Selected Images', id='save-button'),
        dcc.Input(id='save-folder', type='text', placeholder='Enter folder path...', style={'width': '100%', 'marginTop': '5px'}),
        html.Button('Save Story', id='save-story-btn', style={'marginTop': '10px', 'marginLeft': '0px', 'display': 'none'}),
        html.Div(id='save-confirmation', style={'marginTop': '10px'})
    ], style={'width': '42%', 'display': 'inline-block', 'paddingLeft': '3%', 'verticalAlign': 'top'}),

    html.Img(id="hover-thumb", style={
        "display": "none", "position": "fixed", "top": "8px", "left": "8px", "zIndex": 1000,
        "maxWidth": "160px", "maxHeight": "120px", "border": "2px solid #fff", "boxShadow": "0 0 12px #000",
        "backgroundColor": "#000", "objectFit": "contain"
    }),
])

# Show/hide relevant controls for each mode
@app.callback(
    [Output('search-box', 'style'),
     Output('num-images', 'style'),
     Output('story-box', 'style'),
     Output('main-action-btn', 'children')],
    Input('mode-select', 'value'),
)
def toggle_inputs(mode):
    if mode == 'prompt':
        return (
            {'display': 'block', 'width': '60%', 'marginRight': '10px'},
            {'display': 'block', 'width': '15%', 'marginRight': '10px'},
            {'display': 'none', 'width': '70%', 'height': '70px', 'marginRight': '10px'},
            'Search'
        )
    else:
        return (
            {'display': 'none', 'width': '60%', 'marginRight': '10px'},
            {'display': 'none', 'width': '15%', 'marginRight': '10px'},
            {'display': 'block', 'width': '70%', 'height': '70px', 'marginRight': '10px'},
            'Generate Story'
        )
@app.callback(
    [Output('image-display', 'children'),
     Output('scatter-plot', 'figure'),
     Output('save-story-btn', 'style'),
     Output('story-cache', 'data')],
    [Input('main-action-btn', 'n_clicks'),
     Input('scatter-plot', 'clickData'),
     Input('mode-select', 'value'),
     Input('dataset-dropdown', 'value')],
    [State('search-box', 'value'),
     State('num-images', 'value'),
     State('scatter-plot', 'relayoutData'),
     State('story-box', 'value')]
)
def update_images(n_action, clickData, mode, dataset_value, search_value, num_images, relayoutData, story_value):
    if not dataset_value:
        return [], {}, {'display': 'none'}, {}
    latent_name, dim = dataset_value.split("::")
    db_name = latent_name
    n_dim = int(dim)
    df = load_data(latent_name, n_dim=n_dim)
    index, idx2path = load_index(db_name)
    is_3d = all(c in df.columns for c in ['x', 'y', 'z'])
    color_seq = px.colors.qualitative.Dark24
    if is_3d:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='label',
            custom_data=['path', 'thumbnail'],
            color_discrete_sequence=color_seq,
        )
    else:
        fig = px.scatter(
            df, x='x', y='y',
            color='label',
            custom_data=['path', 'thumbnail'],
            color_discrete_sequence=color_seq,
        )
    fig.update_traces(marker=dict(size=4 if is_3d else 8))
    fig.update_layout(
        plot_bgcolor='#121212', paper_bgcolor='#121212', font=dict(color='white'),
        scene=dict(
            xaxis=dict(backgroundcolor="#121212", color="white"),
            yaxis=dict(backgroundcolor="#121212", color="white"),
            zaxis=dict(backgroundcolor="#121212", color="white"),
        ) if is_3d else {}
    )
    trigger = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
    images = []
    story_chunks = []
    story_images = []
    show_save_story = {'display': 'none'}
    story_cache = {}

    # --- Generate Story Mode ---
    if mode == 'story' and trigger == 'main-action-btn' and story_value:
        story_chunks = [chunk.strip() for chunk in story_value.split('\n') if chunk.strip()]
        story_images = []
        for idx, chunk in enumerate(story_chunks):
            results = search(index, idx2path, chunk, 1)
            if results:
                key, path, distance = results[0]
                img_str = encode_image(path)
                story_images.append({'text': chunk, 'path': path, 'img_str': img_str})
        if story_images:
            coords = []
            story_texts = []
            for s in story_images:
                row = df[df["path"] == s["path"]]
                story_texts.append(s['text'])
                if is_3d:
                    coords.append((row['x'].values[0], row['y'].values[0], row['z'].values[0]))
                else:
                    coords.append((row['x'].values[0], row['y'].values[0]))
            if is_3d:
                xs, ys, zs = zip(*coords)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines+markers",
                    line=dict(color="gold", width=4),
                    marker=dict(size=10, color="gold"),
                    text=story_texts,
                    hovertemplate="%{text}<extra></extra>",
                    name="Story Path"
                ))
            else:
                xs, ys = zip(*coords)
                fig.data = tuple(trace for trace in fig.data if getattr(trace, 'name', None) != "Story Path")
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    line=dict(color="gold", width=4),
                    marker=dict(size=14, color="gold"),
                    text=story_texts,
                    hovertemplate="%{text}<extra></extra>",
                    name="Story Path",
                    legendgroup="storypath",
                    showlegend=True
                ))
                fig.add_annotation(
                    x=xs[0], y=ys[0],
                    text="Beginning", showarrow=False,
                    font=dict(size=14, color="gold"),
                    yshift=28, bgcolor="rgba(0,0,0,0.5)", borderpad=6
                )
        for img in story_images:
            images.append(html.Div([
                html.H5(img['text'], style={'marginBottom': '4px', 'color': '#ffc107'}),
                html.Img(src=f'data:image/jpeg;base64,{img["img_str"]}', style={'width': '100%', 'marginBottom': '10px'}),
                daq.BooleanSwitch(id={'type': 'select-image', 'index': img['path']}, on=True, style={'display': 'none'}),
            ], style={'marginBottom': '24px', 'padding': '10px', 'backgroundColor': '#1e1e1e', 'borderRadius': '5px'}))
        show_save_story = {'display': 'block', 'marginTop': '10px'}
        story_cache = {"story": story_images, "chunks": story_chunks}
    # --- Prompt Search Mode ---
    elif mode == 'prompt' and trigger == 'main-action-btn' and search_value:
        results = search(index, idx2path, search_value, num_images)
        highlighted_df = df.loc[[r[0] for r in results]]
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=highlighted_df['x'], y=highlighted_df['y'], z=highlighted_df['z'],
                mode='markers', marker=dict(color='cyan', size=8, symbol='x'),
                name='Search Results'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=highlighted_df['x'], y=highlighted_df['y'],
                mode='markers', marker=dict(color='cyan', size=12, symbol='x'),
                name='Search Results'
            ))
        for key, path, distance in results:
            img_str = encode_image(path)
            images.append(html.Div([
                html.Img(src=f'data:image/jpeg;base64,{img_str}', style={'width': '100%', 'marginBottom': '10px'}),
                daq.BooleanSwitch(id={'type': 'select-image', 'index': path}, on=False),
                html.Span(f'Distance: {distance:.3f}', style={'marginLeft': '10px'})
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#1e1e1e', 'borderRadius': '5px'}))
        show_save_story = {'display': 'none'}
    # --- Scatterplot click, add selected image ---
    elif trigger == 'scatter-plot' and clickData:
        point = clickData['points'][0]
        image_path = point['customdata'][0]
        img_str = encode_image(image_path)
        images.append(html.Div([
            html.Img(src=f'data:image/jpeg;base64,{img_str}', style={'width': '100%', 'marginBottom': '10px'}),
            daq.BooleanSwitch(id={'type': 'select-image', 'index': image_path}, on=False)
        ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#1e1e1e', 'borderRadius': '5px'}))
        show_save_story = {'display': 'none'}

    if is_3d and relayoutData and "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=relayoutData["scene.camera"])

    return images, fig, show_save_story, story_cache

@app.callback(
    Output("hover-thumb", "src"),
    Output("hover-thumb", "style"),
    Input("scatter-plot", "hoverData")
)
def update_hover_thumb(hoverData):
    if hoverData and "points" in hoverData:
        thumb = hoverData["points"][0]["customdata"][1]
        style = {
            "display": "block", "position": "fixed", "top": "100px", "left": "100px", "width": "128px", "height": "128px",
                        "border": "2px solid #fff",
            "zIndex": 1000,
            "boxShadow": "0 0 12px #000"
        }
        return thumb, style
    return dash.no_update, {"display": "none"}

@app.callback(
    Output('save-button', 'style'),
    Input('mode-select', 'value')
)
def toggle_save_selected_button(mode):
    if mode == 'prompt':
        return {'marginTop': '10px', 'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('save-confirmation', 'children'),
    [Input('save-button', 'n_clicks'), Input('save-story-btn', 'n_clicks')],
    [State({'type': 'select-image', 'index': dash.ALL}, 'on'),
     State({'type': 'select-image', 'index': dash.ALL}, 'id'),
     State('save-folder', 'value'),
     State('mode-select', 'value'),
     State('story-cache', 'data')]
)
def save_images(n_clicks_images, n_clicks_story, selections, ids, folder, mode, story_cache):
    msg = ""
    triggered = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None

    # Choose folder based on mode and button
    if triggered == 'save-button':
        # Selections go to output/selections/[your_subfolder]
        subfolder = folder or "session"
        save_dir = os.path.join(SELECTIONS_DIR, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        selected_paths = [id['index'] for id, selected in zip(ids, selections) if selected]
        for path in selected_paths:
            basename = os.path.basename(path)
            cv2.imwrite(os.path.join(save_dir, basename), cv2.imread(path))
        msg = f"{len(selected_paths)} images saved successfully to {save_dir}."
    elif triggered == 'save-story-btn':
        # Stories go to output/stories/[your_subfolder]
        subfolder = folder or "story"
        save_dir = os.path.join(STORIES_DIR, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        if story_cache and 'story' in story_cache:
            for i, item in enumerate(story_cache['story']):
                basename = f"{i:02d}_" + os.path.basename(item['path'])
                cv2.imwrite(os.path.join(save_dir, basename), cv2.imread(item['path']))
            with open(os.path.join(save_dir, "story.txt"), 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(story_cache['chunks']):
                    f.write(f"{i+1}. {chunk}\n")
            msg = f"Story and {len(story_cache['story'])} images saved successfully to {save_dir}."
        else:
            msg = "No story to save."
    return msg

if __name__ == '__main__':
    app.run(debug=False)

