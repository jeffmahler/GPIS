class RenderedImage:
    def __init__(self, image, cam_pos, cam_rot, cam_interest_pt):
        self.image = image
        self.cam_pos = cam_pos
        self.cam_rot = cam_rot
        self.cam_interest_pt = cam_interest_pt
        self.descriptors = {}
