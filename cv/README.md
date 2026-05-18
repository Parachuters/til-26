# CV

Your CV challenge is to detect and classify objects in an image.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/cv` route on port 5002. It is a JSON document structured as such:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    },
    ...
  ]
}
```

The `b64` key of each object in the `instances` list contains the base64-encoded bytes of the input image in JPEG format. The length of the `instances` list is variable.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        [
            {
                "bbox": [left, top, width, height],
                "category_id": category_id
            },
            ...
        ],
        ...
    ]
}
```

where `left` and `top` are the zero-indexed pixel coordinates of the top-left corner of the predicted bounding box, `width` and `height` are the box dimensions in pixels, and `category_id` is the 0-indexed target category. This is LTWH format, not normalized YOLO center format.

If your model detects no objects in a scene, your handler should output an empty list for that scene.

The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where n is the number of input instances. The length of `predictions` must equal that of `instances`.

## Training workflow

Run this on the GCP Workbench instance where `/home/jupyter/$TEAM_TRACK/cv` exists:

```bash
pip install -r cv/requirements.txt
python cv/prepare_dataset.py --data-dir /home/jupyter/$TEAM_TRACK/cv --output-dir /home/jupyter/$TEAM_TRACK/cv_yolo
python cv/train_yolo.py --data /home/jupyter/$TEAM_TRACK/cv_yolo/data.yaml
til build cv
til test cv
```

`cv/train_yolo.py` trains `yolo11l.pt` at `imgsz=1280` for up to 100 epochs with the advanced-track augmentations, then copies `runs/cv/yolo11l_adv/weights/best.pt` to `cv/best.pt` for the Dockerfile.

If inference is too slow during `til test cv`, rebuild with test-time augmentation disabled:

```bash
CV_TTA=0 til build cv
CV_TTA=0 til test cv
```
