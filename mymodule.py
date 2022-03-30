import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import cv2
import math
import pickle


GT_path = '/Users/Jang/Desktop/camscale/GT.csv'
dataframe = pd.read_csv(GT_path)
error_history = []
error_outlier = []
error_outlier_count = []

class Step_detection:
    def __init__(self, participant, trial):

        self.participant = participant
        self.trial = trial
        self.frame_root = osp.join(participant, trial, 'json')
        self.frame_list = [file for file in os.listdir(self.frame_root) if file.endswith(r'.json')]
        self.frame_list.sort()
        self.keypoints = []

        ####Left toe information#####
        self.left_toe = []
        self.d_left_toe = []
        self.d2_left_toe = []
        self.left_toestrike = []
        self.left_toeoff = []
        ####Left heel information#####
        self.left_heel = []
        self.d_left_heel = []
        self.d2_left_heel = []
        self.left_heelstrike = []
        self.left_heeloff = []
        ####Right toe information#####    
        self.right_toe = []
        self.d_right_toe = []
        self.d2_right_toe = []
        self.right_toestrike = []
        self.right_toeoff = []
        ####Right heel information#####
        self.right_heel = []
        self.d_right_heel = []
        self.d2_right_heel = []
        self.right_heelstrike = []
        self.right_heeloff = []

    def findNearNum(self, list, value): # list에서 가장 가까운 값 찾기
            return min(list, key=lambda x:abs(x-value))

    def load_data(self, df): # GT data 가져오기

        GT_data = []
        self.GT = []

        word1 = self.participant.split('_')[1]
        word2 = 'T'+self.trial.split('_')[0]
        word3 = self.trial.split('_')[1]

        for i in range(df.shape[0]-1):
            if df.values[i,0] == word1+'_'+word2+'_'+word3:
                for j in range(len(df.values[i,1:])):
                    if math.isnan(float(df.values[i,j+1])): continue
                    GT_data.append(int(df.values[i,j+1]))

        if len(GT_data) > 0:
            index1 = (GT_data[0] + GT_data[1])/2
            self.GT.append(index1)
            index2 = (GT_data[2] + GT_data[3])/2
            self.GT.append(index2)
            index3 = (GT_data[4] + GT_data[5])/2
            self.GT.append(index3)

            if len(GT_data) == 8:
                index4 = (GT_data[6] + GT_data[7])/2
                self.GT.append(index4)

    def get_keypoints(self): # keypoints 값 가져오기

        for frame_num in self.frame_list:
            frame = osp.join(self.frame_root, frame_num)
            
            with open(frame, 'r') as f:
                frame_data = json.load(f)

            if len(frame_data['people']) == 1:
                self.keypoints.append(frame_data['people'][0]['pose_keypoints_2d'])
            else:
                self.keypoints.append(0)

    
    def PerspectiveTransform(self): # 좌표 변환

        pts1 = np.float32([[150, 225], [460, 240], [10, 370], [550, 410]])
        pts2 = np.float32([[200,0], [800, 0], [200, 600], [800, 600]])   

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        for i in range(len(self.keypoints)):
            target = []
            left_toe = []
            left_heel = []
            right_toe = []
            right_heel = []

            if self.keypoints[i] == 0: continue
            
            ####Left toe#####
            left_toe.append(self.keypoints[i][57])
            if self.keypoints[i][58] < 100: # (Tranform 할 때 y값이 0인 경우 이상한 좌표로 변환되어) 해당 keypoint의 y 값이 100보다 낮게 인식된 경우 y값을 300으로 설정 
                left_toe.append(300)
            else:
                left_toe.append(self.keypoints[i][58])

            ####Left heel#####
            left_heel.append(self.keypoints[i][63])
            if self.keypoints[i][64] < 100:
                left_heel.append(300)
            else:
                left_heel.append(self.keypoints[i][64])

            ####Right toe#####
            right_toe.append(self.keypoints[i][66])
            if self.keypoints[i][67] < 100:
                right_toe.append(300)
            else:
                right_toe.append(self.keypoints[i][67])

            ####Right heel#####
            right_heel.append(self.keypoints[i][72])
            if self.keypoints[i][73] < 100:
                right_heel.append(300)
            else:
                right_heel.append(self.keypoints[i][73])

            target.append(left_toe)
            target.append(left_heel)
            target.append(right_toe)
            target.append(right_heel)

            pts = np.float32(target)
            pts = np.float32([pts])
            transformed = cv2.perspectiveTransform(pts, matrix)


            self.keypoints[i][57] = transformed[0][0][0]
            self.keypoints[i][58] = transformed[0][0][1]
            self.keypoints[i][63] = transformed[0][1][0]
            self.keypoints[i][64] = transformed[0][1][1]
            self.keypoints[i][66] = transformed[0][2][0]
            self.keypoints[i][67] = transformed[0][2][1]
            self.keypoints[i][72] = transformed[0][3][0]
            self.keypoints[i][73] = transformed[0][3][1]


    def get_steps(self):

        for i in range(len(self.keypoints)):

            if self.keypoints[i] == 0: # 만약 해당 프레임에 keypoint가 없는 경우(대개 초반이나 마지막 프레임에서 발생)

                if len(self.left_toe) > 0:
                    if self.left_toe[-1] > 500: self.left_toe.append(1000) # 오른쪽에서 왼쪽으로 걷는 경우 해당 프레임의 x 값을 1000으로 가정
                    else: self.left_toe.append(0) # 왼쪽에서 오른쪽으로 걷는 경우 해당 프레임의 x 값을 0으로 가정
                    if self.left_heel[-1] > 500: self.left_heel.append(1000)
                    else: self.left_heel.append(0)
                    if self.right_toe[-1] > 500: self.right_toe.append(1000)
                    else: self.right_toe.append(0)
                    if self.right_heel[-1] > 500: self.right_heel.append(1000)
                    else: self.right_heel.append(0)

                else:
                    self.left_toe.append(0)
                    self.left_heel.append(0)
                    self.right_toe.append(0)
                    self.right_heel.append(0)

            else: # 해당 프레임에 keypoint가 존재하는 경우(대부분의 경우)

                if self.keypoints[i][57] == 0 or self.keypoints[i][59] < 0.2: # x 값이 0으로 인식되었거나 그 정확도가 0.2보다 낮은 경우
                    if len(self.left_toe) > 0: self.left_toe.append(self.left_toe[-1]) # 만약 직전 값이 존재하면 그 직전 값으로 대체
                    else: self.left_toe.append(0) # 직전 값이 존재하지 않는 경우 0값을 그대로 대입 (초반 프레임에서 발생 가능)
                else: self.left_toe.append(self.keypoints[i][57]) # 아니면 그대로 그 값을 대입
                
                if self.keypoints[i][63] == 0 or self.keypoints[i][65] < 0.2:
                    if len(self.left_heel) > 0: self.left_heel.append(self.left_heel[-1])
                    else: self.left_heel.append(0)
                else: self.left_heel.append(self.keypoints[i][63])
                
                if self.keypoints[i][66] == 0 or self.keypoints[i][68] < 0.2:
                    if len(self.right_toe) > 0: self.right_toe.append(self.right_toe[-1])
                    else: self.right_toe.append(0) 
                else: self.right_toe.append(self.keypoints[i][66])

                if self.keypoints[i][72] == 0 or self.keypoints[i][74] < 0.2:
                    if len(self.right_heel) > 0: self.right_heel.append(self.right_heel[-1])
                    else: self.right_heel.append(0)
                else: self.right_heel.append(self.keypoints[i][72])

                    
        for i in range(len(self.left_toe)-1):
            if self.left_toe[i] == 0 and self.left_toe[i+1]!=0: # 만약 특정 프레임에서 x 값이 0이고 그 다음 값은 0이 아니면 그 값으로 대체
                for j in range(i+1):
                    if self.left_toe[j] == 0:
                        self.left_toe[j] = self.left_toe[i+1]
                break

        for i in range(len(self.left_heel)-1):
            if self.left_heel[i] == 0 and self.left_heel[i+1]!=0:
                for j in range(i+1):
                    if self.left_heel[j] == 0:
                        self.left_heel[j] = self.left_heel[i+1]
                break

        for i in range(len(self.right_toe)-1):
            if self.right_toe[i] == 0 and self.right_toe[i+1]!=0:
                for j in range(i+1):
                    if self.right_toe[j] == 0:
                        self.right_toe[j] = self.right_toe[i+1]
                break

        for i in range(len(self.right_heel)-1):
            if self.right_heel[i] == 0 and self.right_heel[i+1]!=0:
                for j in range(i+1):
                    if self.right_heel[j] == 0:
                        self.right_heel[j] = self.right_heel[i+1]
                break


    def detect_transition(self, tau = 7):

        ####Left toe####
        for i in range(len(self.left_toe)-1):
            self.d2_left_toe.append(self.left_toe[i+1]-self.left_toe[i]) # original 값으로 이계도함수 구하기

            if len(self.d2_left_toe) > 1:
                if np.abs(np.abs(self.d2_left_toe[-2]) - np.abs(self.d2_left_toe[-1])) > 110: # 기울기의 변화(이계도함수)가 너무 큰 값은 주변 값으로 대체(에러값으로 추정)
                    self.left_toe[i+1] = self.left_toe[i]

        self.left_toe_g = gaussian_filter1d(self.left_toe, 2)

        for i in range(len(self.left_toe)-1):
            self.d_left_toe.append(self.left_toe_g[i+1] - self.left_toe_g[i]) # filtered 값으로 도함수 구하기

        for i in range(len(self.d_left_toe)):
            if i > 10: # 10번 째 프레임 이후부터 (초반 프레임 변화는 무시)
                previous = np.abs(self.d_left_toe[i-1])
                present = np.abs(self.d_left_toe[i])

                if  previous > tau and present < tau:  # 기울기 값이 threshold(tau)보다 작아지는 순간 toe strike로 인식
                    self.left_toestrike.append(i)
            
                if previous < tau and present > tau: # 기울기 값이 threshold(tau)보다 커지는 순간 toe off로 인식
                    self.left_toeoff.append(i)

        ####Left heel####
        for i in range(len(self.left_heel)-1):
            self.d2_left_heel.append(self.left_heel[i+1]-self.left_heel[i])

            if len(self.d2_left_heel) > 1:
                if np.abs(np.abs(self.d2_left_heel[-2]) - np.abs(self.d2_left_heel[-1])) > 110:
                    self.left_heel[i+1] = self.left_heel[i]

        self.left_heel_g = gaussian_filter1d(self.left_heel, 2)

        for i in range(len(self.left_heel)-1):
            self.d_left_heel.append(self.left_heel_g[i+1] - self.left_heel_g[i])

        for i in range(len(self.d_left_heel)):
            if i > 10:
                previous = np.abs(self.d_left_heel[i-1])
                present = np.abs(self.d_left_heel[i])

                if  previous > tau and present < tau:
                    self.left_heelstrike.append(i)
            
                if previous < tau and present > tau:
                    self.left_heeloff.append(i)

        ####right toe####
        for i in range(len(self.right_toe)-1):
            self.d2_right_toe.append(self.right_toe[i+1]-self.right_toe[i])

            if len(self.d2_right_toe) > 1:
                if np.abs(np.abs(self.d2_right_toe[-2]) - np.abs(self.d2_right_toe[-1])) > 110:
                    self.right_toe[i+1] = self.right_toe[i]

        self.right_toe_g = gaussian_filter1d(self.right_toe, 2)

        for i in range(len(self.right_toe)-1):
            self.d_right_toe.append(self.right_toe_g[i+1] - self.right_toe_g[i])

        for i in range(len(self.d_right_toe)):
            if i > 10:
                previous = np.abs(self.d_right_toe[i-1])
                present = np.abs(self.d_right_toe[i])

                if  previous > tau and present < tau:
                    self.right_toestrike.append(i)
            
                if previous < tau and present > tau:
                    self.right_toeoff.append(i)

        ####right heel####
        for i in range(len(self.right_heel)-1):
            self.d2_right_heel.append(self.right_heel[i+1]-self.right_heel[i])

            if len(self.d2_right_heel) > 1:
                if np.abs(np.abs(self.d2_right_heel[-2]) - np.abs(self.d2_right_heel[-1])) > 110:
                    self.right_heel[i+1] = self.right_heel[i]

        self.right_heel_g = gaussian_filter1d(self.right_heel, 2)

        for i in range(len(self.right_heel)-1):
            self.d_right_heel.append(self.right_heel_g[i+1] - self.right_heel_g[i])

        for i in range(len(self.d_right_heel)):
            if i > 10:
                previous = np.abs(self.d_right_heel[i-1])
                present = np.abs(self.d_right_heel[i])

                if  previous > tau and present < tau:
                    self.right_heelstrike.append(i)
            
                if previous < tau and present > tau:
                    self.right_heeloff.append(i)


    def get_ROI(self):
        
        self.ROI1 = [] # left heelstrike로 찾은 것
        self.ROI2 = [] # right heelstrike로 찾은 것

        if self.left_heel[0] < 500: # 왼족에서 오른쪽으로 걷는 경우

            for i in range(max(len(self.left_heelstrike), len(self.right_heelstrike))):

                if len(self.left_heelstrike) > i: #i 번째 left heelstrike 존재여부
                    pair1 = self.findNearNum(self.right_toeoff, self.left_heelstrike[i]) # 해당 left heelstrike와 가장 가까운 right toeoff 찾기
                    if np.abs(self.left_heelstrike[i] - pair1) < 10: # 해당 right toeoff가 10 프레임 이내 존재하는가? (너무 가까우면 에러 값으로 추정)
                        index = (self.left_heelstrike[i] + pair1)/2 # left heelstrike와 right toeoff 평균 점을 index로 잡기
                        if self.left_heel_g[int(index)] > 315 and self.left_heel_g[int(index)] < 890: # 관찰 좌표 범위 안에 들어오는가?
                            if len(self.ROI1) > 0:   # ROI1에 값이 존재하는가?
                                if np.abs(self.ROI1[-1]- index) > 14:   # 직전 ROI1의 값과 차이가 14 프레임을 초과하는가? (너무 가까우면 에러 값으로 추정)
                                    if len(self.ROI2) > 0: # ROI2에 값이 존재하는가?
                                        if np.abs(self.ROI2[-1]- index) > 5: # 직전 ROI2의 값과 차이가 5 프레임을 초과하는가? (너무 가까우면 에러 값으로 추정)
                                            self.ROI1.append(index)
                                    else:
                                        self.ROI1.append(index)
                            else:
                                self.ROI1.append(index)

                if len(self.right_heelstrike) > i:
                    pair2 = self.findNearNum(self.left_toeoff, self.right_heelstrike[i])
                    if np.abs(self.right_heelstrike[i] - pair2) < 10:
                        index = (self.right_heelstrike[i] + pair2)/2
                        if self.right_heel_g[int(index)] > 315 and self.right_heel_g[int(index)] < 890:
                            if len(self.ROI2) > 0:
                                if np.abs(self.ROI2[-1]- index) > 14:
                                    if len(self.ROI1) > 0:
                                        if np.abs(self.ROI1[-1]- index) > 5:
                                            self.ROI2.append(index)
                                    else:
                                        self.ROI2.append(index)
                            else:
                                self.ROI2.append(index)

        else: # 오른쪽에서 왼쪽으로 걷는 경우

            for i in range(max(len(self.left_heelstrike), len(self.right_heelstrike))):

                if len(self.left_heelstrike) > i:
                    pair1 = self.findNearNum(self.right_toeoff, self.left_heelstrike[i])
                    if np.abs(self.left_heelstrike[i] - pair1) < 10:
                        index = (self.left_heelstrike[i] + pair1)/2
                        if self.left_heel_g[int(index)] > 120 and self.left_heel_g[int(index)] < 750:
                            if len(self.ROI1) > 0:     
                                if np.abs(self.ROI1[-1]- index) > 14:
                                    if len(self.ROI2) > 0:
                                        if np.abs(self.ROI2[-1]- index) > 5:
                                            self.ROI1.append(index)
                                    else:
                                        self.ROI1.append(index)
                            else:
                                self.ROI1.append(index)



                if len(self.right_heelstrike) > i:
                    pair2 = self.findNearNum(self.left_toeoff, self.right_heelstrike[i])
                    if np.abs(self.right_heelstrike[i] - pair2) < 10:
                        index = (self.right_heelstrike[i] + pair2)/2
                        if self.right_heel_g[int(index)] > 120 and self.right_heel_g[int(index)] < 750:
                            if len(self.ROI2) > 0:
                                if np.abs(self.ROI2[-1]- index) > 14:
                                    if len(self.ROI1) > 0:
                                        if np.abs(self.ROI1[-1]- index) > 5:
                                            self.ROI2.append(index)
                                    else:
                                        self.ROI2.append(index)
                            else:
                                self.ROI2.append(index)

        self.ROI = []

        for i in range(len(self.ROI1)):
            self.ROI.append([self.ROI1[i], self.left_heel_g[int(self.ROI1[i])]])
        
        for i in range(len(self.ROI2)):
            self.ROI.append([self.ROI2[i], self.right_heel_g[int(self.ROI2[i])]])

        self.ROI.sort(key=lambda x:abs(x[1]-500)) # x 값이 500(적외선 카메라 범위)에 가까운 것을 기준으로 4개의 ROI 지정
        self.ROI = self.ROI[:4]
        self.ROI.sort(key=lambda x:x[0])

        for i in range(len(self.ROI)):
            self.ROI[i] = self.ROI[i][0]



    def plot_steps(self):

        for i in range(len(self.GT)):
            if i == 0:
                plt.axvline(x = self.GT[i], linestyle = 'dashed', label = 'GT')
            else:
                plt.axvline(x = self.GT[i], linestyle = 'dashed')
            plt.text(self.GT[i], 640, self.GT[i], fontsize='10')

        for i in range(len(self.ROI)):
            if i == 0:
                plt.axvline(x = self.ROI[i], color='r', linestyle = 'dashed', label = 'prediction')
            else:
                plt.axvline(x = self.ROI[i], color='r', linestyle = 'dashed')
            plt.text(self.ROI[i], 660, self.ROI[i], fontsize='10')

        ####Left toe####
        plt.plot(range(len(self.left_toe_g)), self.left_toe_g, label='Left toe')
        plt.plot(self.left_toeoff, self.left_toe_g[self.left_toeoff], "o")

        ####Left heel####
        plt.plot(range(len(self.left_heel_g)), self.left_heel_g, label='Left heel')
        plt.plot(self.left_heelstrike, self.left_heel_g[self.left_heelstrike], "o")

        ####Right toe####
        plt.plot(range(len(self.right_toe_g)), self.right_toe_g, label='Right toe')
        plt.plot(self.right_toeoff, self.right_toe_g[self.right_toeoff], "o")

        ####Right heel####
        plt.plot(range(len(self.right_heel_g)), self.right_heel_g, label='Right heel')
        plt.plot(self.right_heelstrike, self.right_heel_g[self.right_heelstrike], "o")
        
        participant = self.participant.split('_')[1]
        plt.title(participant+'_T'+str(self.trial))
        plt.xlabel('frames')
        plt.ylabel('x coordinate')
        plt.legend(fontsize=10)
        plt.show()


    def get_errors(self):
        
        self.error = 0
        
        for i in range(len(self.GT)):
            pair = self.findNearNum(self.ROI, self.GT[i]) # prediction과 가장 가까운 GT를 기준으로 error 측정
            self.error += np.abs(self.GT[i] - pair) / 16

        self.error = self.error / len(self.GT)
        error_history.append(self.error)

        participant = self.participant.split('_')[1]

        if self.error > 0.15:
            error_outlier.append({'participant' : participant, 'trial' : self.trial, 'error': self.error, 'nth': len(error_history)-1})

        print(f'Average Error time(s) of {participant}_T{self.trial} = {self.error:.3f}s')

def implement_all():
    file_path = '/Users/Jang/Desktop/camscale/poses/individual/'
    file_list = [file for file in os.listdir(file_path) if not file.endswith(r'Store')]
    file_list.sort()
    pred_dict, gt_dict = {}, {}

    for file in file_list:
        file2_path = osp.join(file_path, file)
        file2_list = [file for file in os.listdir(file2_path) if not file.endswith(r'Store')]
        file2_list.sort()

        for file2 in file2_list:

            if file == '0330_P30' and file2 == '1_10': continue # GT값 없음

            Trial = Step_detection(file_path+file, file2)
            Trial.get_keypoints()
            Trial.PerspectiveTransform()
            Trial.load_data(dataframe)
            if len(Trial.GT) == 0: continue 
            Trial.get_steps()
            Trial.detect_transition()
            Trial.get_ROI()
            Trial.get_errors()

            pred_dict[file.split('_')[1]+'_T'+file2] = Trial.ROI
            gt_dict[file.split('_')[1]+'_T'+file2] = Trial.GT


    with open('/Users/Jang/Desktop/camscale/pred_dict.pkl', 'wb') as f:
        pickle.dump(pred_dict, f)

    with open('/Users/Jang/Desktop/camscale/gt_dict.pkl', 'wb') as f:
        pickle.dump(gt_dict, f)

    mean = np.mean(error_history)
    std = np.std(error_history)

    print(f'mean = {mean: .3f}, standard deviation = {std: .3f}')

    plt.bar(range(len(error_history)), error_history, label = 'Error')

    for i in range(len(error_outlier)):
        plt.text(error_outlier[i]['nth'], error_outlier[i]['error'], str(error_outlier[i]['participant'])+'_T'+str(error_outlier[i]['trial'])+'='+str(round(error_outlier[i]['error'],3))+'s', fontsize='5')

    plt.axhline(y = mean, color = 'r', label = f'mean = {mean: .3f}')
    plt.plot([], [], label = f'std = {std: .3f}')
    plt.title('Average errors')
    plt.xlabel('Sample')
    plt.ylabel('Error(s)')
    plt.legend(fontsize=10)
    plt.show()


def implement_one(participant, trial):
    file_path = '/Users/Jang/Desktop/camscale/poses/individual/'
    Trial = Step_detection(file_path+participant, trial)
    Trial.get_keypoints()
    Trial.PerspectiveTransform()
    Trial.load_data(dataframe)
    Trial.get_steps()
    Trial.detect_transition()
    Trial.get_ROI()
    Trial.plot_steps()
    Trial.get_errors()


def read_pickle():
    with open('/Users/Jang/Desktop/camscale/pred_dict.pkl', 'rb') as f:
        pred = pickle.load(f)
    for key in pred.keys():
        print(key,':',pred[key])

    # with open('/Users/Jang/Desktop/camscale/gt_dict.pkl', 'rb') as f:
    #     gt = pickle.load(f)
    # for key in gt.keys():
    #     print(key,':',gt[key])



#### Implementation ####


implement_all() # 해당 함수를 이용하면 GT가 존재하는 모든 sample에 대해서 error, mean, std값을 그래프로 보여줍니다.
# implement_one('0404_P38', '4_10') # 해당 함수를 이용하면 하나의 sample에 대해서 step의 x좌표의 변화와 ROI prediction 및 GT값을 그래프로 볼 수 있습니다.
# read_pickle()



