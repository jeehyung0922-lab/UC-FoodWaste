import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from assets import HW, build_bundle, load_bundle, load_probability, match_maps, rasterize, val_rows, write, VERSION, SPLIT_SHA256
from engine import METADATA_DECLARATION, verify_metadata
from metrics import auspro, pixel_metrics, region_metric, select, sweep, sweep_regions


class EvaluatorTests(unittest.TestCase):
    def test_perfect_background_foreground(self):
        gt=np.array([0,1])
        for probability,expected in [
            ([1,1,0,0],[2,0,0,2]),([0,0,0,0],[0,0,2,2]),([1,1,1,1],[2,2,0,0])]:
            n,_=sweep(np.array(probability),gt,[gt],np.array([.5]))
            self.assertEqual(n[0].tolist(),expected)
        self.assertEqual(pixel_metrics(2,0,0,2)["dice"],1)

    def test_10_and_5_percent(self):
        self.assertEqual(region_metric(np.array([2,1]),np.array([20,20])),(1,.5,.75))

    def test_touching_and_overlapping_polygons(self):
        a=rasterize([2,2,20,2,20,20,2,20],"a")
        b=rasterize([20,2,38,2,38,20,20,20],"b")
        c=rasterize([12,12,30,12,30,30,12,30],"c")
        self.assertEqual((a % 768).max() + 1, (b % 768).min())
        self.assertEqual(len(np.intersect1d(a,b)),0)
        self.assertGreater(len(np.intersect1d(a,c)),0)
        p=np.zeros(512*768,dtype=np.float32)
        p[a]=.9
        _,hits=sweep(p,np.union1d(a,c),[a,c],np.array([.8]))
        self.assertEqual(int(hits[0][0]),len(a))
        self.assertEqual(int(hits[1][0]),len(np.intersect1d(a,c)))

    def test_zero_area(self):
        with self.assertRaisesRegex(ValueError,"zero-area"):
            rasterize([0,0,4,4,8,8],"line")

    def test_empty_and_undefined(self):
        self.assertEqual(region_metric(np.empty(0),np.empty(0)),(0,None,None))
        background=pixel_metrics(0,0,0,10)
        for key in ("pixel_recall","pixel_precision","iou","dice"):
            self.assertIsNone(background[key])
        self.assertEqual(background["pixel_fpr"],0)
        self.assertEqual(background["pixel_accuracy"],1)
        self.assertIsNone(pixel_metrics(2,0,0,0)["pixel_fpr"])

    def test_selection_order(self):
        grid=np.array([0.,.3,.6,.8,.9])
        counts=np.array([[1,3,0,97],[1,2,0,98],[1,2,0,98],[1,2,0,98],[1,1,0,99]])
        self.assertEqual(select(grid,counts,np.array([3,2,2,2,1]),np.array([1.,.7,.8,.8,.9]),.02),3)

    def test_auspro_duplicate_and_boundary(self):
        x=np.array([0,0,.01,.03]);y=np.array([0,.4,.6,1])
        self.assertAlmostEqual(auspro(x,y,.02),.6)
        self.assertIsNone(auspro(x,np.full(4,np.nan),.02))
        self.assertAlmostEqual(auspro(np.array([.02,0,.02]),np.array([.5,.2,.8]),.02),.5)
        self.assertIsNone(auspro(x,y,.04))

    def test_spro_dominates_region_recall(self):
        areas=np.array([99,100,101])
        hits=np.array([[0,9,10,99],[0,9,10,100],[0,9,10,101]])
        detected,spro=sweep_regions(hits,areas,np.arange(4))
        self.assertTrue(np.all(spro>=detected/len(areas)))
        for i in range(4):
            self.assertAlmostEqual(spro[i],region_metric(hits[:,i],areas)[2])

    def test_operator_metadata_requires_a_real_method(self):
        with self.assertRaisesRegex(ValueError,"geometric alignment"):
            verify_metadata(dict(METADATA_DECLARATION,probability_method=" "))
        declaration=dict(METADATA_DECLARATION,probability_method="Channel 1 softmax on aligned grid")
        self.assertEqual(verify_metadata(declaration),declaration)

    def test_optimized_matches_reference(self):
        rng=np.random.default_rng(17)
        p=rng.choice(np.array([0,.1,.25,.5,1],np.float32),size=100)
        gt=np.flatnonzero(rng.random(100)<.4)
        mask=np.isin(np.arange(100),gt)
        regions=[np.arange(25,50),np.arange(35,80)]
        grid=np.r_[np.linspace(0,1,1001,dtype=np.float32),np.nextafter(np.float32(1),np.float32(2))]
        counts,hits=sweep(p,gt,regions,grid)
        for j,t in enumerate(grid):
            pred=p>=t
            self.assertEqual(counts[j].tolist(),[int(x.sum()) for x in
                [pred&mask,pred&~mask,~pred&mask,~pred&~mask]])
            self.assertEqual([r[j] for r in hits],[pred[r].sum() for r in regions])

    def test_map_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            images=[dict(id=1,filename="image000001")]
            with self.assertRaisesRegex(ValueError,"Missing"): match_maps(p,images)
            (p/"1.npy").touch(); (p/"image000001.npy").touch()
            with self.assertRaisesRegex(ValueError,"Duplicate"): match_maps(p,images)
            (p/"1.npy").unlink(); (p/"image999999.npy").touch()
            with self.assertRaisesRegex(ValueError,"outside"): match_maps(p,images)

    def test_map_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"sample.npy"
            for array,reason in [(np.zeros((2,2),np.float32),"expected"),
                 (np.zeros(HW,np.float64),"expected"),(np.full(HW,np.nan,np.float32),"NaN"),
                 (np.full(HW,np.inf,np.float32),"Inf"),(np.full(HW,1.5,np.float32),"outside")]:
                np.save(path,array)
                with self.assertRaisesRegex(ValueError,reason): load_probability(path)
            np.save(path,np.full(HW,.25,np.float32))
            self.assertEqual(load_probability(path).shape,(HW[0]*HW[1],))
            with path.open("wb") as stream:
                np.savez(stream, probability=np.zeros(HW,np.float32))
            with self.assertRaisesRegex(ValueError,"NPZ archive"): load_probability(path)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"sample.npy: unreadable"): load_probability(path)

    def test_wrong_split_and_gt_has_no_polygon_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            split=p/"split.csv"
            split.write_text("image_id,split\n1,val\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"approved"):
                val_rows(split)
            path=p/"source.csv"
            path.write_text("Image ID,Image Name,Width,Height,Polygons\n1,Image000001.npy,1536,1024,[]\n2,ignored,0,0,DO_NOT_PARSE\n",encoding="utf-8")
            mask=np.zeros(HW,np.uint8)
            mask[2:6,2:6]=1
            Image.fromarray(mask).save(p/"image000001.png")
            with patch("assets.val_rows",return_value={1:{"filename":"image000001","group":"A","has_contaminant":"1"}}):
                with self.assertRaisesRegex(ValueError,"emptiness"):
                    build_bundle(split,p,path,p/"assets")
            self.assertFalse((p/"assets").exists())

    def test_asset_and_metadata_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"validation_pixels.npz").write_bytes(b"changed")
            write(p/"region_manifest.json",{})
            write(p/"asset_manifest.json",dict(version=VERSION,split="val",split_sha256=SPLIT_SHA256,
                  pixel_archive_sha256="original",region_manifest_sha256="unused"))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                load_bundle(p)
            write(p/"input_metadata.json",dict(positive_class="Background"))
            with self.assertRaisesRegex(ValueError,"Contaminant"):
                verify_metadata(p/"input_metadata.json")


if __name__=="__main__":
    unittest.main()
