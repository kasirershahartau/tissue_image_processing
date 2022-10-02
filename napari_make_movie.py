from napari_animation import Animation

def scale_zchannel(viewer,new_zscale):
    for channel_num in range(len(viewer.layers)):
        if len(viewer.layers[channel_num].data.shape) == 4:
            viewer.layers[channel_num].scale = [1, new_zscale, 1, 1]

def make_movie(viewer, output_path):
    viewer.dims.set_point(0, 0)
    viewer.dims.ndisplay = 3

    animation = Animation(viewer)
    animation.capture_keyframe()

    image = viewer.layers[0].data
    viewer.dims.set_point(0, image.shape[0])

    animation.capture_keyframe(steps=image.shape[0])
    animation.animate(output_path, fps=10)