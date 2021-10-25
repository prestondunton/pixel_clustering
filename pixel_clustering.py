import cv2 as cv
import io
from itertools import chain
import math
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

MAX_K = 10
RANDOM_SEED = 440


def hex_to_float_tuple(hex_string):
    hex_string = hex_string.lstrip('#')
    return tuple(int(hex_string[i:i + 2], 16) for i in (0, 2, 4))


def float_tuple_to_hex(float_tuple):
    return '#%02x%02x%02x' % tuple(int(f) for f in float_tuple)


def row_to_rgb_tuple(row):
    rgb_tuple = tuple([int(value) for index, value in row.items() if index in ['R', 'G', 'B']])
    return 'rgb' + str(rgb_tuple)


# ===================================================================================================================


def render_introduction():
    st.write('Modern image processing and computer vision focuses on convolution and other machine learning methods, '
             'which allow computers to model very complex patterns and shapes in images.  However, this doesn\'t mean '
             'that simpler models can\'t provide any useful information to us.  This app demonstrates how K Means '
             'clustering can be used to gain insight into images, and be used to create some cool artwork.')

    st.write('One simple analysis we can do on images is to describe the distribution of pixels in space.  Because '
             'each pixel has 3 values associated to it, RGB, we can think of each pixel living in 3D space, where each '
             'axis corresponds to the value (0-255) of R, G, and B respectively.  RGB is the canonical way to process '
             'images because computers store images in RGB, but there are many other spaces in which colors can also '
             'be described.  HSV (Hue, Saturation, Value) and HSL (Hue, Saturation, Light) are examples of these other '
             'spaces.  Whereas RGB models color in a cubical space, these spaces model in a cylindrical space based on '
             'the color wheel.')

    columns = st.columns(3)
    with columns[0]:
        st.image('./images/rgb_space.png', caption='RGB Color Space (Source: Wikipedia)')

    with columns[1]:
        st.image('./images/hsv_space.png', caption='HSV Color Space (Source: Wikipedia)')

    with columns[2]:
        st.image('./images/hsl_space.png', caption='HSL Color Space (Source: Wikipedia)')

# st.write('HSV and HSL are just two examples of the many different ways people have thought to model colors.  Other '
    #         'examples include YUV, YIQ, YCbCr, and CMYK, and each have an interesting interpretation.  A good '
    #         'challenge is to try and imagine the spatial transformations when converting between one space to '
    #         'another.  Hint: they\'re mostly non-linear transformations.')

    st.write(
        'This distribution of pixels can be described with simple summary statistics, like the mean and standard '
        'deviation of the R, G, and B channels.  However, another more interesting method to describe distributions in '
        'space is clustering.  Where summary statistics only describe one channel at a time, clustering '
        'summarizes the distribution in the entire space all at once.  There are also many different methods of '
        'clustering, which gives us flexibility in our model.')

    st.write('This app uses Scikit-Learn\'s implementation of K Means clustering (with kmeans++  '
             'initialization).  Use the file uploader to upload an image and interactively explore the distribution '
             'of its pixels!')


@st.cache(show_spinner=False)
def init_file(uploaded_file):
    with st.spinner('Initializing application.  Please wait.'):
        if uploaded_file is not None:

            print(f'Initializing uploaded image "{uploaded_file.name}"')

            image = np.array(Image.open(uploaded_file))
            st.session_state['width'] = image.shape[0]
            st.session_state['height'] = image.shape[1]
            st.session_state['total_pixels'] = image.shape[0] * image.shape[1]

            print(f'\tSetting (width, height) to {(st.session_state["width"], st.session_state["height"])}')

            for i in range(1, MAX_K + 1):
                st.session_state[f'clustering_{i}'] = None

            cluster(image, k=1)
            cluster(image, k=2)
            st.session_state['k'] = 2

            return image


def cluster(image, k):

    if st.session_state[f'clustering_{k}'] is None:
        with st.spinner(f'Clustering image with {k} clusters.  Please wait.'):
            print(f'Clustering image with {k} clusters')

            clustering = KMeans(n_clusters=k)
            clustering.fit(image.reshape((-1, 3)))

            st.session_state[f'clustering_{k}'] = clustering

            print(f'\tAvailable clusterings: {[st.session_state[f"clustering_{i}"] for i in range(1, MAX_K + 1)]}')


def render_distribution_section(image):
    st.header('Clusters in RGB Space')

    k = st.slider(label='# of Clusters', min_value=1, max_value=MAX_K, step=1, value=2)
    st.session_state['k'] = k
    cluster(image, k)

    plot_pixels = st.slider('# of Pixels to Plot', min_value=0, max_value=1000, value=500, step=10)
    num_centroids = st.session_state['k'] if st.checkbox('Show centroids', value=True) else 0
    st.plotly_chart(get_pixel_plot(image, plot_pixels, num_centroids), use_container_width=True)
    st.write('If you look at this plot for enough images, you\'ll notice some similar patterns across images.  '
             'Most photographs tend to have pixels distributed around the line R=G=B, which is a line from the '
             'black corner to the white corner of the RGB cube.  In addition, there are sometimes lines of pixels '
             'that are offset from this line, but still are parallel to R=G=B.  These can be interpreted as hues '
             'in the image that fall under different lighting and shadows.  These patterns provide a good '
             'motivation to also try modeling in HSV or HSL space.')


@st.cache(show_spinner=False)
def get_pixel_plot(image, plot_pixels, k=0):
    with st.spinner('Loading pixel plot.  Please wait.'):
        print(f'Rendering pixel plot with {plot_pixels} pixels and {k} centroids')

        pixels = np.copy(image.reshape((-1, 3)))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(pixels)
        pixels = pixels[0:plot_pixels]

        pixels = pd.DataFrame({'R': pixels[:, 0], 'G': pixels[:, 1], 'B': pixels[:, 2]})
        pixels['size'] = 1
        pixels['Type'] = 'Pixel'

        if k > 0:
            # Add centroids
            for color in st.session_state[f'clustering_{k}'].cluster_centers_:
                pixels = pixels.append({'R': color[0], 'G': color[1], 'B': color[2],
                                        'size': 5, 'Type': 'Centroid'}, ignore_index=True)
            max_size = 15
        else:
            max_size = 7

        pixels['color'] = pixels.apply(row_to_rgb_tuple, axis=1)

        domain = (0, 255)
        fig = px.scatter_3d(pixels, x='R', y='G', z='B',
                            range_x=domain, range_y=domain, range_z=domain,
                            color='color', color_discrete_map='identity',
                            size='size', size_max=max_size,
                            symbol='Type', hover_data={'Type': True, 'R': True, 'G': True, 'B': True, 'size': False})

        camera = dict(eye=dict(x=1.25, y=-1.25, z=1.25))
        fig.update_layout(scene_camera=camera)
        fig.update_layout(scene_aspectmode='cube')

        return fig


def render_tree_section():
    st.header('Cluster Hierarchies')

    st.write('When doing clustering, one common issue is determining how many clusters to use.  There are some '
             'methods that automatically determine how many to use, but usually this is done manually.  Because '
             'we are just doing exploratory analysis for fun, we can try several values and compare them!')

    st.write('In the hierarchy below, you can see how as we add more clusters, the colors are split to '
             'better approximate the original image.  If we draw an edge between each centroid in one clustering '
             'and the closest centroid in the previous clustering, we can see that some centroids don\'t really '
             'change, while others split in opposite directions.  You can think of this as mixing paint in reverse.  '
             'Each time we add a new centroid, the colors have an opportunity to un-mix into their original colors, '
             'and K Means will actually un-mix the color that will benefit most from this process.')

    st.write('**Change the number of clusters using the slider above to add nodes to this tree!**')

    clusterings = [st.session_state[f'clustering_{i}']
                   for i in range(1, MAX_K + 1)
                   if st.session_state[f'clustering_{i}'] is not None]
    nodes, edges, config = get_centroid_tree(clusterings)
    agraph(nodes=nodes, edges=edges, config=config)

    st.write('In this animated version of the hierarchy, we show how the centroids move in RGB space as we add more '
             'of them.  Pause the animation to explore different perspectives in the space.')
    st.plotly_chart(get_centroid_animation(clusterings, edges), use_container_width=True)


@st.cache(show_spinner=False)
def get_centroid_tree(clusterings):
    with st.spinner('Loading centroid hierarchy.  Please wait.'):
        print('Rendering centroid tree')

        edges = create_edges(clusterings)
        nodes = create_nodes(edges)

        config = Config(width=860,
                        height=86 * len(clusterings[-1].cluster_centers_),
                        directed=True,
                        collapsible=False,
                        panAndZoom=False,
                        staticGraph=True,
                        node={'labelProperty': 'label', 'renderLabel': False},
                        link={'labelProperty': 'label', 'renderLabel': False, 'fontSize': 14}
                        )

        return nodes, edges, config


def create_edges(clusterings):
    print('Creating tree edges')
    edges = []

    for i in range(1, len(clusterings)):
        edges += [Edge(source=edge[0], target=edge[1], label='{0:.1f}'.format(edge[2]))
                  for edge in get_edge_tuples(clusterings[i - 1], clusterings[i])]

    edges.sort(key=lambda x: x.source)

    return edges


def get_edge_tuples(clustering1, clustering2, debug=False):
    centroid_pairs, distances = get_centroid_pairs(clustering1, clustering2, debug)

    edges = []

    for i in range(len(centroid_pairs)):
        edges.append((f'{len(clustering1.cluster_centers_)}_{centroid_pairs[i][0]}',
                      f'{len(clustering2.cluster_centers_)}_{centroid_pairs[i][1]}',
                      distances[centroid_pairs[i][0]][centroid_pairs[i][1]]))

    return edges


def get_centroid_pairs(clustering1, clustering2, debug=False):
    # if clustering1 has more centers than cluster 2
    if clustering1.cluster_centers_.shape[0] > clustering2.cluster_centers_.shape[0]:
        # swap clusterings
        clustering1, clustering2 = clustering2, clustering1

    centroid_pairs = []

    start_d = distance_matrix(clustering1.cluster_centers_, clustering2.cluster_centers_)
    d = np.copy(start_d)

    while len(centroid_pairs) < d.shape[0]:

        if debug:
            print(f'Distance Matrix \n\t{d}')

        closest_centers = tuple([int(value) for value in np.unravel_index(d.argmin(), d.shape)])

        centroid_pairs.append(closest_centers)

        d[closest_centers[0], :] = 9999999999
        d[:, closest_centers[1]] = 9999999999

    if debug:
        print(f'Distance Matrix \n\t{d}')

    # allows us to index correctly later using clustering2.labels_
    centroid_pairs.sort(key=lambda x: x[1])

    if debug:
        print(f'Centroid Pairs: \n\t{centroid_pairs}')

    for i in range(0, len(clustering2.cluster_centers_)):
        if i >= len(centroid_pairs) or centroid_pairs[i][1] != i:
            closest_center = (np.argmin(start_d[:, i]), i)

            centroid_pairs.insert(i, tuple([int(value) for value in closest_center]))

    if debug:
        print(f'Centroid Pairs: \n\t{centroid_pairs}')

    return centroid_pairs, start_d


def create_nodes(edges):
    node_size = 750

    node_ids = []
    previous_row_ordering = None
    for k in range(1, MAX_K + 1):
        if st.session_state[f'clustering_{k}'] is not None:
            if previous_row_ordering is not None:
                current_row_ordering = [[edge.target for edge in edges if edge.source == node] for node in
                                        previous_row_ordering]
                current_row_ordering = list(chain.from_iterable(current_row_ordering))
                node_ids += current_row_ordering
            else:
                current_row_ordering = ['1_0']

            previous_row_ordering = current_row_ordering

    x_cords = []
    for k in range(1, MAX_K + 1):
        if st.session_state[f'clustering_{k}'] is not None:
            x_cords += [i * node_size / MAX_K for i in range(k)]

    x_cords = x_cords[1:]
    x_cords = [x + 15 for x in x_cords]

    nodes = []
    if st.session_state['clustering_1'] is not None:
        nodes.append(Node(id='1_0', size=node_size, renderLabel=False,
                          x=15, y=(node_size / MAX_K),
                          color=float_tuple_to_hex(st.session_state[f'clustering_{1}'].cluster_centers_[0]))
                     )

    for i in range(len(node_ids)):
        parts = node_ids[i].split('_')
        k = int(parts[0])
        j = int(parts[1])

        nodes.append(Node(id=f'{k}_{j}', size=node_size, renderLabel=False,
                          x=x_cords[i], y=(k * node_size / MAX_K),
                          color=float_tuple_to_hex(st.session_state[f'clustering_{k}'].cluster_centers_[j])))

    return nodes


@st.cache(show_spinner=False)
def get_centroid_animation(clusterings, edges):
    with st.spinner('Loading hierarchy animation.  Please wait.'):
        clustering_ks = [len(clusterings[i].cluster_centers_)
                         for i in range(len(clusterings))
                         if clusterings[i] is not None]

        print(f'Rendering centroid animation for clusterings {clustering_ks}')

        frames = create_animation_frames(clustering_ks, edges, steps_per_frame=50)

        frames['color'] = frames.apply(row_to_rgb_tuple, axis=1)
        frames['size'] = 5

        domain = (0, 255)
        fig = px.scatter_3d(frames, x='R', y='G', z='B',
                            range_x=domain, range_y=domain, range_z=domain,
                            color='color', color_discrete_map='identity',
                            size='size', size_max=20,
                            animation_frame='frame',
                            symbol='centroid', symbol_sequence=['circle'],
                            hover_data={'R': True, 'G': True, 'B': True, 'frame': False})

        duration = 5
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = duration
        camera = dict(eye=dict(x=1.25, y=-1.25, z=1.25))

        fig.update_layout(scene_camera=camera)
        fig.update_layout(scene_aspectmode='cube')

        return fig


def create_animation_frames(clustering_ks, edges, steps_per_frame):
    print('Creating animation frames')
    frames = pd.DataFrame(columns=['R', 'G', 'B', 'frame', 'centroid'])

    leaf_edges = [edge for edge in edges if edge.target not in [edge.source for edge in edges]]

    for leaf_edge in leaf_edges:
        frames_recorded = 0

        leaf_k = int(leaf_edge.target.split('_')[0])
        leaf_i = int(leaf_edge.target.split('_')[1])

        for k in clustering_ks:
            if k < leaf_k:

                current_edge = leaf_edge
                ancestor_k = int(current_edge.source.split('_')[0])
                ancestor_i = int(current_edge.source.split('_')[1])

                while ancestor_k != k:
                    # get the parent, traverse backwards through branches of the tree
                    current_edge = [edge for edge in edges if edge.target == current_edge.source][0]
                    ancestor_k = int(current_edge.source.split('_')[0])
                    ancestor_i = int(current_edge.source.split('_')[1])

                frame_color = st.session_state[f'clustering_{ancestor_k}'].cluster_centers_[ancestor_i]

                frames = frames.append({'R': frame_color[0], 'G': frame_color[1], 'B': frame_color[2],
                                        'frame': frames_recorded, 'centroid': leaf_edge.target},
                                       ignore_index=True)
                frames_recorded += 1

        frame_color = st.session_state[f'clustering_{leaf_k}'].cluster_centers_[leaf_i]
        frames = frames.append({'R': frame_color[0], 'G': frame_color[1], 'B': frame_color[2],
                                'frame': frames_recorded, 'centroid': leaf_edge.target},
                               ignore_index=True)

    frames = interpolate_frames(frames, steps_per_frame)

    return frames


def interpolate_frames(frames, steps_per_frame):
    new_frames = pd.DataFrame(columns=['R', 'G', 'B', 'frame', 'centroid'])

    frames['frame'] *= steps_per_frame
    max_frame = frames['frame'].max()

    unique_centroids = frames['centroid'].unique()

    for centroid in unique_centroids:
        for i in range(max_frame):
            correct_centroid = (frames['centroid'] == centroid)
            start_frame = (frames['frame'] == steps_per_frame * (math.floor(i / steps_per_frame)))
            end_frame = (frames['frame'] == steps_per_frame * (math.ceil(i / steps_per_frame)))

            start_color = frames[correct_centroid & start_frame].to_numpy()[:, 0:3]

            end_color = frames[correct_centroid & end_frame].to_numpy()[:, 0:3]

            step_color = ((end_color - start_color) * ((i % steps_per_frame) / steps_per_frame)) + start_color
            step_color = step_color[0]

            new_frames = new_frames.append({'R': step_color[0], 'G': step_color[1], 'B': step_color[2],
                                            'frame': i, 'centroid': centroid}, ignore_index=True)

    return new_frames


def render_color_pickers():
    k = st.session_state['k']

    print(f'Rendering {k} color pickers')

    columns = st.columns(k)

    colors = []

    for i in range(k):
        with columns[i]:
            colors.append(st.color_picker(
                f'Cluster {i}',
                float_tuple_to_hex(
                    tuple(st.session_state[f'clustering_{k}'].cluster_centers_[i])
                )
            ))

    return colors


@st.cache(show_spinner=False)
def get_quantized_image(k, colors):

    with st.spinner('Loading quantized image. Please wait.'):

        print(f'Rendering quantized image with {k} colors')

        rgb_colors = np.array([hex_to_float_tuple(c) for c in colors])

        quantized_image = rgb_colors[st.session_state[f'clustering_{k}'].labels_]
        quantized_image = quantized_image.reshape(st.session_state['width'],
                                                  st.session_state['height'], 3).astype('uint8')

        return quantized_image


def render_quantized_image(k, colors, filename):

    quantized_image = get_quantized_image(k, colors)
    
    st.image(quantized_image,
             caption=f'Color Quantization with {k} Clusters', use_column_width=True)

    is_success, buffer = cv.imencode(".jpg", quantized_image[:, :, ::-1])
    io_buf = io.BytesIO(buffer)
    quantized_jpg = io_buf.read()

    output_filename = filename.strip('.jpg') + '_quantized.jpg'

    st.download_button('Download Image', quantized_jpg, file_name=output_filename)


# ===================================================================================================================


def render_app():
    st.set_page_config(page_title='Pixel Clustering', page_icon=':large_blue_circle:')

    st.title('Pixel Clustering with K Means')
    st.subheader('By Preston Dunton')

    render_introduction()

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg'])
    image = init_file(uploaded_file)

    if image is not None:
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        render_distribution_section(image)

        render_tree_section()

        st.header('Color quantization')
        st.write('Now for the fun part!  If we reassign each pixel\'s color to the color of its nearest centroid, we '
                 'can style our image.  Depending on how the clustering comes out, you can get some pop-art style '
                 'versions of the original image.')

        st.write('**Use the color pickers to style your image and save it using the download button below!**')
        colors = render_color_pickers()
        render_quantized_image(st.session_state['k'], colors, uploaded_file.name)


render_app()
