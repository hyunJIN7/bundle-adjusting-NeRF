# process_arkit_data.py

### 사용법
코드 작성 시 경로 이상하게 설정해서 data 폴더로 이동 후 아래 명령어 실행

`python process_arkit_data.py --expname data_name` 

data_name엔 data/arkit/ 하위에 원하는 데이터 넣어 놓고 실행.

1. m4v 파일에서 image frame 추출 후 ARpose.txt 와 sync 맞춰 SyncedPose.txt 파일 생성 
2. [right,up,back] --> [right,forward,up]로 축 변환 후 key frame selection 과정  (frame 파일 마다 저장 포맷 다름)
3. select된 키 프레임에서 train : val = 0.9 : 0.1 비율로 이미지와 포즈 저장. 순서서대로 [0:0.9], [0.9:1] 나눔
4. test는 sync 맞춤 데이터에서 랜덤 추출
5.  `iphone_train_val_images` 폴더는 train+val 합쳐진 데이터

### For each target, provide relevant utilities to evaluate our system.

- process_arkit_data.py : [right,up,back] 원본대로 로드 후 키프레임 셀랙 할때만 [right,forward,up]로 진행하고 pose 저장은  [right,up,back]
- process_arkit_data_frame2.py :[right,up,back] process_arkit_data_frame2 거친후  [right,forward,up]
- process_arkit_data_frame3.py : [right,up,back] 그대로 저장하되 keframe select은 every 30 frames for opti-track (선택 프레임 주기 변)

### Dataset format
- {train,val,test}_transformation.txt : `timestamp {train,val,test}/image_name r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
- SyncedPoses.txt : `timestamp imagenum(string) tx ty tz(m) qx qy qz qw`
[ios_logger](https://github.com/Varvrar/ios_logger) 참고

