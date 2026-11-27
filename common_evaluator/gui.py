"""Tkinter interface for the locked Common Evaluator."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import sys
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from engine import METADATA_DECLARATION, evaluate


def asset_location():
    base = Path(sys.executable if getattr(sys,"frozen",False) else __file__).resolve().parent
    return base/"validation_assets"


class App:
    def __init__(self, root):
        self.root=root
        self.events=queue.Queue()
        self.maps=tk.StringVar()
        self.model=tk.StringVar()
        self.status=tk.StringVar(value="확률맵(.npy) 84장이 있는 폴더를 선택하세요.")
        root.title("UC-FoodWaste Common Evaluator v0.1")
        panel=ttk.Frame(root,padding=18)
        panel.grid(sticky="nsew")
        ttk.Label(panel,text="UC-FoodWaste | Provisional Validation",font=("Segoe UI",15,"bold")).grid(sticky="w",columnspan=3,pady=(0,12))
        ttk.Label(panel,text="확률맵 폴더").grid(row=1,column=0,sticky="w")
        ttk.Entry(panel,textvariable=self.maps,width=57).grid(row=1,column=1,sticky="ew",padx=8)
        ttk.Button(panel,text="폴더 선택",command=self.browse).grid(row=1,column=2)
        ttk.Label(panel,text="모델 이름").grid(row=2,column=0,sticky="w",pady=8)
        ttk.Entry(panel,textvariable=self.model).grid(row=2,column=1,sticky="ew",padx=8)
        self.start=ttk.Button(panel,text="평가 시작",command=self.start_evaluation)
        self.start.grid(row=3,column=0,columnspan=2,sticky="w",pady=6)
        self.open=ttk.Button(panel,text="결과 폴더 열기",command=self.open_results,state="disabled")
        self.open.grid(row=3,column=2)
        self.bar=ttk.Progressbar(panel,maximum=85)
        self.bar.grid(row=4,column=0,columnspan=3,sticky="ew",pady=(6,4))
        ttk.Label(panel,textvariable=self.status,wraplength=615).grid(row=5,column=0,columnspan=3,sticky="w")
        self.results=tk.Text(panel,width=78,height=9,state="disabled",wrap="word")
        self.results.grid(row=6,column=0,columnspan=3,sticky="ew",pady=(8,0))
        self.output=None
        root.resizable(False,False)

    def browse(self):
        folder=filedialog.askdirectory(title="Select Validation probability map folder")
        if folder:
            self.maps.set(folder)
            if not self.model.get():
                self.model.set(Path(folder).name)

    def start_evaluation(self):
        if not self.maps.get().strip():
            messagebox.showerror("입력 오류","확률맵 폴더를 선택하세요.")
            return
        maps=Path(self.maps.get().strip())
        if not maps.is_dir():
            messagebox.showerror("입력 오류",f"폴더를 찾을 수 없습니다.\n{maps}")
            return
        assets=asset_location()
        if not (assets/"asset_manifest.json").is_file():
            messagebox.showerror("Validation 자산 없음",f"실행 파일 옆에 검증된 validation_assets 폴더를 두세요.\n{assets}")
            return
        declaration=None
        if not (maps/"input_metadata.json").exists():
            method=simpledialog.askstring(
                "확률맵 생성 정보 확인",
                "Contaminant의 임계값 적용 전 확률이며, 고정 Validation GT와 공간 정렬이 확인된 경우에만 입력하세요.\n"
                "실제 클래스 매핑, Sigmoid/Softmax 방식, 해상도 및 공간 정렬 절차를 적으세요.",
                parent=self.root)
            if not method or not method.strip():
                return
            declaration=dict(METADATA_DECLARATION,probability_method=method.strip())
        stamp=datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        out=Path.home()/"UC-FoodWaste Evaluations"/stamp
        self.begin(self.model.get().strip() or maps.name,maps,assets,out,declared_metadata=declaration)

    def begin(self, model, maps, assets, out, metadata=None, on_complete=None, *, declared_metadata=None):
        self.start.config(state="disabled")
        self.open.config(state="disabled")
        self.bar["value"]=0
        self.status.set("Validating the asset pack and model inputs...")
        self.on_complete=on_complete
        def worker():
            try:
                result=evaluate(model,maps,assets,out,metadata,
                    progress=lambda done,total:self.events.put(("progress",(done,total))),
                    declared_metadata=declared_metadata)
                self.events.put(("success",(out,result)))
            except Exception as exc:
                self.events.put(("error",str(exc)))
        Thread(target=worker,daemon=True).start()
        self.root.after(80,self.poll)

    def poll(self):
        while True:
            try:
                kind,value=self.events.get_nowait()
            except queue.Empty:
                self.root.after(80,self.poll)
                return
            if kind=="progress":
                done,total=value
                self.bar.configure(maximum=total,value=done)
                self.status.set(f"Processed {min(done,84)} of 84 Validation images.")
                continue
            self.start.config(state="normal")
            if kind=="success":
                self.output,result=value
                self.status.set(f"Saved to {self.output}")
                self.open.config(state="normal")
                fields=("region_recall","iou","pixel_recall","pixel_precision","pixel_fpr","spro","auspro")
                lines=[f"Threshold: {result['selected_threshold']:.6g}"]
                for key in fields:
                    metric=result[key]
                    lines.append(f"{key}: {metric:.4%}" if metric is not None else f"{key}: N/A")
                self.results.config(state="normal")
                self.results.delete("1.0","end")
                self.results.insert("end","\n".join(lines))
                self.results.config(state="disabled")
            else:
                self.status.set(value)
                if not self.on_complete:
                    messagebox.showerror("평가 중단",value)
            if self.on_complete:
                self.on_complete(kind,value)
            return

    def open_results(self):
        try:
            os.startfile(self.output)
        except OSError as exc:
            messagebox.showerror("결과 폴더 열기 실패",f"{self.output}\n{exc}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    # Maintainer-only harness exercises real Tk event loop and the same worker.
    parser.add_argument("--smoke-test",nargs=4,type=Path,metavar=("MAPS","META","ASSETS","OUT"))
    parser.add_argument("--smoke-declaration",help=argparse.SUPPRESS)
    args=parser.parse_args()
    root=tk.Tk()
    app=App(root)
    status=0
    if args.smoke_test:
        maps,metadata,assets,out=args.smoke_test
        def complete(kind,value):
            nonlocal status
            status=0 if kind=="success" else 1
            if status:
                out.parent.mkdir(parents=True,exist_ok=True)
                (out.parent/"gui_smoke_error.txt").write_text(value,encoding="utf-8")
            root.destroy()
        declaration=(dict(METADATA_DECLARATION,probability_method=args.smoke_declaration)
                     if args.smoke_declaration else None)
        root.after(50,lambda:app.begin("Frozen GUI Smoke",maps,assets,out,metadata,complete,
                                      declared_metadata=declaration))
    root.mainloop()
    return status


if __name__=="__main__":
    raise SystemExit(main())
