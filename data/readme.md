# process_arkit_data.py

사용법
코드 작성 시 경로 이상하게 설정해서 data 폴더로 이동 후 아래 명령어 실행

`python process_arkit_data.py --expname data_name` 

data_name엔 data/arkit/ 하위에 원하는 데이터 넣어 놓고 실행.

1. m4v 파일에서 image frame 추출 후 ARpose.txt 와 sync 맞춰 SyncedPose.txt 파일 생성 
2. [right,up,back] --> [right,forward,up]로 축 변환 후 key frame selection 과정 
3. select된 키 프레임에서 train : val = 0.9 : 0.1 비율로 이미지와 포즈 저장. 순서서대로 [0:0.9], [0.9:1] 나눔
4. test는 sync 맞춤 데이터에서 랜덤 추출
5.  `iphone_train_val_images` 폴더는 train+val 합쳐진 데이터


