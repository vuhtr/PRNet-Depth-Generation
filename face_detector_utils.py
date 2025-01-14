import cv2

def get_bbox(detections, image_shape, MARGIN=0.02):
    max_area = 0
    bbox = None

    for detection in detections:
        bbox = detection.location_data.relative_bounding_box
        x = max(0, bbox.xmin - MARGIN)
        y = max(0, bbox.ymin - MARGIN)
        w = min(1, bbox.width + 2 * MARGIN)
        h = min(1, bbox.height + 2 * MARGIN)

        x = int(x * image_shape[1])
        y = int(y * image_shape[0])
        w = int(w * image_shape[1])
        h = int(h * image_shape[0])

        area = w * h
        if area > max_area:
            max_area = area
            bbox = [detection.score[0], x, y, w, h]

    return bbox

def translate_bbox(bbox):
    # return left, right, top, bototm (startX, endX, startY, endY)
    return bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]


def detect_face(image, face_detection):
    '''
        image: (H, W, C)
        face_detection: face detection model (MediaPipe)

        Return: is_flip_image, bbox of main face
    '''

    # fip image upside down
    image_flip = cv2.flip(image, 0)

    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results_flip = face_detection.process(cv2.cvtColor(image_flip, cv2.COLOR_BGR2RGB))

    # detect for both image and image_flip
    bbox = [0] * 5
    bbox_flip = [0] * 5
    if results.detections is not None:
        bbox = get_bbox(results.detections, image.shape)
    if results_flip.detections is not None:
        bbox_flip = get_bbox(results_flip.detections, image.shape)

    # get the result with better score
    final_bbox = bbox
    flip = False
    if bbox_flip[0] > bbox[0]:
        final_bbox = bbox_flip
        # flip the bbox (score, x, y, w, h)
        # final_bbox[2] = image.shape[0] - final_bbox[2] - final_bbox[4]
        flip = True
                                     
    if final_bbox[0] == 0:
        return None

    final_bbox = final_bbox[1:]
    return flip, translate_bbox(final_bbox)

    
    
    