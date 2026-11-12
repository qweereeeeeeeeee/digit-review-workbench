"""Geometry and provenance checks; recognition quality is recorded separately."""
import pytest
from objects.general import box_to_original,iou,verify_bundle


def test_crop_coordinates_are_restored_to_original():
    assert box_to_original([10,20,90,80],(100,100),(400,200),(100,50))==[.275,.35,.475,.65]

@pytest.mark.parametrize('box',[[0,0,0,40],[40,40,10,10],[0,0,float('nan'),20]])
def test_invalid_generated_boxes_are_not_drawn(box):
    assert box_to_original(box,(100,100),(100,100),(0,0)) is None


def test_clip_boxes_and_overlap():
    assert box_to_original([-10,-10,120,120],(100,100),(100,100),(0,0))==[0,0,1,1]
    assert iou([0,0,1,1],[0,0,1,1])==1
    assert iou([0,0,.1,.1],[.5,.5,1,1])==0


def test_pinned_official_weights_and_code():
    assert verify_bundle('florence','florence-source.json')['revision']=='f6c1a25888ffc1d945ee8a1a77ac833c7303d46e'
    assert len(verify_bundle('ppocr','ppocr-source.json')['files'])==4
