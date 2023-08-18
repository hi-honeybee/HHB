# 실행방법
(급하게 수정하느라 코드가 정리되있지 않을 수 있습니다)


datasets 폴더를 만들고 그 밑에 해당하는 날짜로 폴더를 만들고 그 밑에 영상.mp4 파일을 넣어주세요

예시) datasets/0802/1.mp4


HHB.py에서 def cycle(date='date'): 의 'data'를 폴더명으로 적어주세요

예시) def cycle(date='0802'):


python HHB.py를 실행하면 track_result/0802/1.pickle로 결과가 저장됩니다.




# Hi HoneyBee! pipeline
모듈 단위로 수정할 수 있는 하이허니비 파이프라인입니다.

자신이 구현한 코드를 올린 뒤에 refer_path.yaml에서 해당 파일의 경로를 추가해 주면 됩니다.

* weight와 dataset은 용량 때문에 안올라가서 드라이브에서 따로 받으시기 바랍니다

## 코드 구현 시 유의사항

* HHB.py 내의 코드는 상의가 이뤄진 후에 수정하도록 합니다.

* 통일성 및 명시성을 위해 사용하려는 코드는 Module.HHB_func(arg, **kargs) 형태로 호출하기로 정합니다.
예시:

```python
box = detector.HHB_detect(frame,**kargs)
```

모듈간의 의존성을 줄이기 위해 input과 output은 가능한 주석에 명시된 대로 따르도록 합니다.
다만, 경우에 따라 유연하게 수정해서 사용해도 됩니다.

* kargs는 모든 모듈에서 공통으로 접근하므로 명확히 구분 가능한 키워드를 사용하도록 합니다.


## 모듈 기능

**DataLoader.HHB_dataload** : path로부터 data(video)를 읽고 frame을 반환하는 iterator 반환

**DetectorLoader.HHB_detectorload** : detection 모델을 반환

**TrackerLoader.HHB_trackerload** : tracking 모델을 반환

**Iprp.HHB_imgpreprocess** : 이미지 전처리

**detector.HHB_detect** : 이미지로부터 대상 위치를 예측, box에 대한 tensor 반환

**Dpop.HHB_boxpostprocess** : detection 결과 후처리

**tracker.HHB_track** : box로부터 추적, 추적한 물체의 STrack 반환

**Tpop.HHB_trackpostprocess** : tracking 결과 후처리

**Visualizer.HHB_visualize** : tracking 결과 시각화 

**Save.HHB_save** : tracking 결과 저장
