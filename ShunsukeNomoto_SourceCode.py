from cmath import inf
from statistics import variance
import cv2
import numpy as np
from matplotlib import pyplot as pyp
import math
from scipy.stats import norm
import copy

try:

    # 画像サイズ
    rows = 4000
    cols = 6000

    # サンプル数(値は50程度で十分)
    sample = 50

    # 使用する画像のインポート(環境によって都度ファイル指定)
    img1 = cv2.imread('./Python/HDR/SourcePicture/DSC00325.jpg')
    img2 = cv2.imread('./Python/HDR/SourcePicture/DSC00326.jpg')
    img3 = cv2.imread('./Python/HDR/SourcePicture/DSC00327.jpg')
    img4 = cv2.imread('./Python/HDR/SourcePicture/DSC00328.jpg')
    img5 = cv2.imread('./Python/HDR/SourcePicture/DSC00329.jpg')
    img6 = cv2.imread('./Python/HDR/SourcePicture/DSC00331.jpg')
    img7 = cv2.imread('./Python/HDR/SourcePicture/DSC00332.jpg')
    img8 = cv2.imread('./Python/HDR/SourcePicture/DSC00333.jpg')
    img9 = cv2.imread('./Python/HDR/SourcePicture/DSC00334.jpg')
    img10 = cv2.imread('./Python/HDR/SourcePicture/DSC00335.jpg')
    img11 = cv2.imread('./Python/HDR/SourcePicture/DSC00336.jpg')
    img12 = cv2.imread('./Python/HDR/SourcePicture/DSC00337.jpg')
    img13 = cv2.imread('./Python/HDR/SourcePicture/DSC00338.jpg')
    img14 = cv2.imread('./Python/HDR/SourcePicture/DSC00339.jpg')
    img15 = cv2.imread('./Python/HDR/SourcePicture/DSC00340.jpg')
    img16 = cv2.imread('./Python/HDR/SourcePicture/DSC00341.jpg')
    img17 = cv2.imread('./Python/HDR/SourcePicture/DSC00342.jpg')
    img18 = cv2.imread('./Python/HDR/SourcePicture/DSC00343.jpg')
    img19 = cv2.imread('./Python/HDR/SourcePicture/DSC00344.jpg')
    img20 = cv2.imread('./Python/HDR/SourcePicture/DSC00345.jpg')
    img21 = cv2.imread('./Python/HDR/SourcePicture/DSC00346.jpg')
    img22 = cv2.imread('./Python/HDR/SourcePicture/DSC00347.jpg')
    img23 = cv2.imread('./Python/HDR/SourcePicture/DSC00349.jpg')
    img24 = cv2.imread('./Python/HDR/SourcePicture/DSC00350.jpg')
    img25 = cv2.imread('./Python/HDR/SourcePicture/DSC00351.jpg')
    img26 = cv2.imread('./Python/HDR/SourcePicture/DSC00352.jpg')
    img27 = cv2.imread('./Python/HDR/SourcePicture/DSC00353.jpg')
    img28 = cv2.imread('./Python/HDR/SourcePicture/DSC00354.jpg')
    img29 = cv2.imread('./Python/HDR/SourcePicture/DSC00355.jpg')
    img30 = cv2.imread('./Python/HDR/SourcePicture/DSC00356.jpg')
    img31 = cv2.imread('./Python/HDR/SourcePicture/DSC00357.jpg')
    img32 = cv2.imread('./Python/HDR/SourcePicture/DSC00358.jpg')
    img33 = cv2.imread('./Python/HDR/SourcePicture/DSC00359.jpg')
    img34 = cv2.imread('./Python/HDR/SourcePicture/DSC00362.jpg')
    img35 = cv2.imread('./Python/HDR/SourcePicture/DSC00363.jpg')
    img36 = cv2.imread('./Python/HDR/SourcePicture/DSC00364.jpg')
    img37 = cv2.imread('./Python/HDR/SourcePicture/DSC00365.jpg')
    img38 = cv2.imread('./Python/HDR/SourcePicture/DSC00366.jpg')
    img39 = cv2.imread('./Python/HDR/SourcePicture/DSC00367.jpg')
    img40 = cv2.imread('./Python/HDR/SourcePicture/DSC00368.jpg')
    img41 = cv2.imread('./Python/HDR/SourcePicture/DSC00369.jpg')
    img42 = cv2.imread('./Python/HDR/SourcePicture/DSC00370.jpg')
    img43 = cv2.imread('./Python/HDR/SourcePicture/DSC00371.jpg')
   
#   画像を配列化
    img_list = [img1,img2,img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,img13,img14,img15,img16,img17,img18,img19,img20,img21,img22,img23,img24,img25,img26,img27,img28,img29,img30,img31,img32,img33,img34,img35,img36,img37,img38,img39,img40,img41,img42,img43] 


#シャッタースピード（露出時間exprosure time）をfloatで配列化）
    exporsure_times = np.array([1/4000,1/3200,1/2500,1/2000,1/1600,1/1250,1/1000,1/800,1/640,1/500,1/400,1/320,1/250,1/200,1/160,1/125,1/100,1/80,1/60,1/50,1/40,1/30,1/25,1/20,1/15,1/13,1/10,1/8,1/6,1/5,1/4,1/3,0.4,0.5,0.625,0.8,1,1.3,1.6,2,2.5,3.2,4],dtype=np.float32)

# サンプル箇所を特定(cv2.createCalibrateDebevec関数の内部処理(オープンソース)参照)
    def get_samplepoints():

        sample_point = []

        x_points = int(math.sqrt(int(sample) * cols / rows))

        if 0 < x_points and x_points <= cols:
            y_points = int(sample/x_points)
        
        if (0 < y_points and y_points <= rows):
            step_x = int(cols/x_points)
            step_y = int(rows/y_points)

            from_x = int(step_x/2)
            from_y = int(step_y/2)

            to_x = from_x + step_x * x_points 
            to_y = from_y + step_y * y_points 

        for x in range(from_x,to_x,step_x):
            for y in range(from_y,to_y,step_y):
                if 0<= x and x < cols and 0 <= y and y < rows:
                    # 縦優先だから(row,col)
                    sample_point.append((y,x))

        return sample_point,step_x,step_y


# 新たなサンプリング箇所の設定
    def new_sample_pos(width,radius):
        
        center_pos_list = []
        sample_pos_list = []
        num_r = int(rows / width)
        num_c = int(cols / width)

        f = get_samplepoints()
        img_samplepoint = f[0]

        for c in range(1,num_c,1):
            for r in range(1,num_r,1):
                center_pos_list.append((width*r,width*c))

        for i in img_samplepoint:
            center_pos_list.append(i)

        center_pos_list = set(center_pos_list)

        for i in center_pos_list:
            if (i[0]>= radius and i[1]>=radius) or (i[0]<=rows-radius and i[1]<=cols-radius) :
                sample_pos_list.append(i)

        return sample_pos_list   


# 正規分布から画像中のサンプル箇所（samplepointとは別）に対してその周辺のピクセルの画素値の加重平均の算出
    def pixelvalue_distinction_inimg(img,width,radius):

        weighted_ave_list_inimg = []

        sample_pos_list = new_sample_pos(width,radius)

        for i in sample_pos_list:
            around_pixel_list = []
            # samplepoint(i)を基準に縦横-radiusからradiusの範囲の中で条件を満たす（半径radiusの円内にあるか)を判別
            for w in range((i[0]-radius),(i[0]+radius+1),1): 
                for j in range((i[1]-radius),(i[1]+radius+1),1):
                    # (x-a)**2+(y-b)**2<=radius**2を満たしていれば[(w,j)]
                    if (w-i[0])**2+(j-i[1])**2 <= radius**2:
                        around_pixel = (w,j)
                        around_pixel_list.append(around_pixel) 

            ave_b_list = []
            ave_g_list = []
            ave_r_list = []

            # 重みのリスト
            weighted_val_list = []


            for inside_pixel in around_pixel_list:   
            
                # 1.inside_pixelの中心ピクセル（サンプルポイント）からの距離の取得
                pixel_dis = (math.sqrt((inside_pixel[0]-i[0])**2 + (inside_pixel[1]-i[1])**2))

                # 2.標準正規分布から1.の距離に応じた重みの取得。その値をリストに追加
                weighted_val = norm.pdf(pixel_dis)
                weighted_val_list.append(weighted_val)
    
                # 3.inside_pixelのBGRの値の取得
                val_b = img[inside_pixel[0],inside_pixel[1],0]
                val_g = img[inside_pixel[0],inside_pixel[1],1]
                val_r = img[inside_pixel[0],inside_pixel[1],2]

                # 4.各チャンネルの値*重み
                ave_b = val_b * weighted_val
                ave_g = val_g * weighted_val
                ave_r = val_r * weighted_val

                # 5.4.を各チャンネルごとにリストに追加
                ave_b_list.append(ave_b)
                ave_g_list.append(ave_g)
                ave_r_list.append(ave_r)
            
            weighted_ave_b = 0
            weighted_ave_g = 0
            weighted_ave_r = 0
            weighted_val_sum = 0
            
            # 6.5.から各チャンネルごとの値の和を取り、重みの総和(weighted_val_list)で割る（重みの総和が1ではないため）
            for b in ave_b_list:
                weighted_ave_b += b

            for g in ave_g_list:
                weighted_ave_g += g

            for r in ave_r_list:
                weighted_ave_r += r 

            for weight in weighted_val_list:
                weighted_val_sum += weight

            # 各チャンネルの加重平均
            weighted_ave = [weighted_ave_b/weighted_val_sum,weighted_ave_g/weighted_val_sum,weighted_ave_r/weighted_val_sum]
            weighted_ave_list_inimg.append([i,weighted_ave])

        return weighted_ave_list_inimg
        

# pixelvalue_distinction()から加重平均とサンプルポイントの画素値を比べ、サンプルポイントの条件がいいか判定
    def samplepoint_condition(width,radius):

        original_pos = get_samplepoints()[0]
        original_pos_val = []

        sample_pos = new_sample_pos(width,radius)
        sample_num = len(sample_pos)
        pos_sum_val_eachimg = []
        sum_val_eachpos = [[] for i in range(sample_num)]
        
        for img_num in range(len(img_list)):
            
            img = img_list[img_num]
            pos_sum_val_eachpos = []
            weighted_ave_list_inimg = pixelvalue_distinction_inimg(img,width,radius)

            for i in weighted_ave_list_inimg:
                
                samplepoint_pixelvalue_b = img[i[0][0],i[0][1],0]
                weighted_ave_val_b = i[1][0]

                samplepoint_pixelvalue_g = img[i[0][0],i[0][1],1]
                weighted_ave_val_g = i[1][1]

                samplepoint_pixelvalue_r = img[i[0][0],i[0][1],2]
                weighted_ave_val_r = i[1][2]

                # その点の各チャンネルにおける中心ポイントとの差
                val_diff_b = abs(samplepoint_pixelvalue_b-weighted_ave_val_b)
                val_diff_g = abs(samplepoint_pixelvalue_g-weighted_ave_val_g)
                val_diff_r = abs(samplepoint_pixelvalue_r-weighted_ave_val_r)

                #各チャンネルにおける差
                # val_diff = [val_diff_b,val_diff_g,val_diff_r]
                val_sum = val_diff_b+val_diff_g+val_diff_r

                pos_sum_val_eachpos.append([i[0],val_sum])

            pos_sum_val_eachimg.append(pos_sum_val_eachpos)

        for num in pos_sum_val_eachimg:
            for i in range(sample_num):
                sum_val_eachpos[i].append(num[i])

        sum_total_eachpos = [[i,0] for i in sample_pos]

        for i in range(sample_num):
            for img_num in range(len(img_list)):
                sum_total_eachpos[i][1] += sum_val_eachpos[i][img_num][1]
            if sum_total_eachpos[i][0] in original_pos:
                original_pos_val.append(sum_total_eachpos[i].copy())

        lower_val_pos = []

        sum_total_eachpos = sorted(sum_total_eachpos,key = lambda x:x[1])

        original_pos_val = sorted(original_pos_val,key = lambda x:x[1])
        lower_lim = original_pos_val[len(original_pos_val)-1][1]

        # 元のサンプルポイントの差の加重平均の最も大きい値をlower_limとし、それより大きい値を持つものをsum_total_eachposから削除する
        for i in sum_total_eachpos:
            if i[1] <= lower_lim:
                lower_val_pos.append(i)

            else:
                break          

        print(lower_val_pos)
        
        return lower_val_pos,original_pos_val



# samplepoint_condition(bad_cond,good_cond)で取得したgood_posについて、そのポジションにおける輝度の合計値を調べる
    def check_luminance(width,radius,division_num):

        hsv_img_list = []
        samplepointcondition = samplepoint_condition(width,radius)
        sample_pos = samplepointcondition[0]
        original_pos_val = samplepointcondition[1]
        img_list_copy  = copy.deepcopy(img_list)

        for img in img_list_copy:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hsv_img_list.append(img)

        pos_lumi_val_list = []

        for pos in sample_pos:
            pos_lumi_val = [pos,0]
            for img in hsv_img_list:
                pos_lumi_val[1] += img[pos[0]][2]
            pos_lumi_val_list.append(pos_lumi_val)

        lumi_list = []

        for i in pos_lumi_val_list:
            lumi_list.append(i[1])

        # 最大輝度*画像枚数で輝度値の総和による条件付け
        max_lumi_val = 255 * len(img_list)

        bins_f = []

        # division_numは明るさ0~255までを何分割したか
        for i in range(1,division_num+1,1):
            x = int(i/division_num*max_lumi_val)
            bins_f.append(x)

        # lumi_listに関して、num個のリストに分けて格納する（〜から〜までの範囲でいくつというような、範囲によるばらつきが欲しい）
        separate_lumi_val_list = np.digitize(lumi_list,bins=bins_f)
        separate_pos_lumi_list = [[] for i in range(division_num)]

        for i in range (len(pos_lumi_val_list)):
            for n in range(division_num):
                if n == separate_lumi_val_list[i]:

                    separate_pos_lumi_list[n].append(pos_lumi_val_list[i][0][0])

        return separate_pos_lumi_list,original_pos_val


# samplepoint_conditionで導いた条件のよくないポイントに条件のいいところの画素値を置換する
    def replace_pixel(width,radius,division_num):

        checkluminance = check_luminance(width,radius,division_num)

        original_pos_val = checkluminance[1]
        original_pos = []

    
        for i in original_pos_val:
            original_pos.append(i[0])

        separate_pos_lumi_list = checkluminance[0]

        replacement_point_num = 0
        replacement_point = []

        # num個に分けたリストのそれぞれに対して、一回ずつrandomに抽出してreplacement_pointに入れる作業をreplacement_point_num < len(bad_pos)を満たす間続ける
        while replacement_point_num < len(original_pos):
            for i in range(len(separate_pos_lumi_list)):
                n = len(separate_pos_lumi_list[i])
                if n > 0:
                    x = separate_pos_lumi_list[i][0]
                    replacement_point.append(x)
                    separate_pos_lumi_list[i].remove(x)
                    replacement_point_num += 1     

                else:
                    continue
            
                if replacement_point_num == len(original_pos):
                    break 

        sort_y = sorted(replacement_point, key = lambda x:(x[1],x[0]))

        im_list = copy.deepcopy(img_list)

        pix_val = [[] for i in range(len(img_list))]
        for num in range(len(img_list)):
            for i in sort_y:
                pix_val[num].append(img_list[num][i])

        for num in range(len(im_list)):
            for i in range(len(original_pos)):
                val = pix_val[num][i]
                for x in range(-15,15,1):
                    for y in range(-15,15,1):
                        pos = (original_pos[i][0]+x,original_pos[i][1]+y)
                # pos = (original_pos[i][0],original_pos[i][1])
                        im_list[num][pos] = val

            # (環境によって都度指定)
            filename = "./Python/HDR/Replaceimg_lambda0/Replaced_img/test/im_list" + str(num) + ".jpg"
            cv2.imwrite(filename,im_list[num])


# replace_pixelで作成した新しい画像を読み込み、あらたな画像のリストの作成
    def import_newimg(width,radius,division_num):

        replace_pixel(width,radius,division_num) 

        img_list_replaced = []

        for num in range(len(img_list)):
            # (環境によって都度指定)
            filename = "./Python/HDR/Replaceimg_lambda0/Replaced_img/test/im_list" + str(num) + ".jpg"
            img = cv2.imread(filename)
            img_list_replaced.append(img)
        
        condition = "width" + str(width) + "_num" + str(division_num) + "_radius" + str(radius) + "_test"

        return img_list_replaced,condition


    replace_process = import_newimg(2000,15,1)

    img_list = replace_process[0]
    condition = replace_process[1]


# 反応曲線(かメラレスポンスカーブ)を生成(各輝度値ごとに抽出される))
    def CalibrateDebevec():
        # （cv2.createCalibrateDebevec([samples[,lambda[,random]]])）
        cal_debevec = cv2.createCalibrateDebevec(samples=sample,lambda_=0,random=False)
        crf_debevec = cal_debevec.process(img_list,exporsure_times.copy())
        crf_debevec = crf_debevec.astype(np.float32)


        # グラフ化
        crf_b = crf_debevec[:,:,0]
        crf_g = crf_debevec[:,:,1]
        crf_r = crf_debevec[:,:,2] 

        # crf_debevec[:,:,c]の中でinfもしくは0があったら、その前後の値から補完する(線形補間)
        def check_bgr():  
            
            for c in range(3):
                for i in range(256):  
                    if (crf_debevec[:,:,c][i] == inf and i > 1 and i < 255) or (crf_debevec[:,:,c][i] == 0 and i > 1 and i < 255) or (crf_debevec[:,:,c][i] > 100 and i > 1 and i < 255):                  
                        for num in range(1,255,1):               
                            if crf_debevec[:,:,c][i+num] != 0 and crf_debevec[:,:,c][i+num] != inf and crf_debevec[:,:,c][i+num] < 100:
                                for n in range(num):
                                    crf_debevec[:,:,c][i+n] = crf_debevec[:,:,c][i+n-1] + ((crf_debevec[:,:,c][i+num] - crf_debevec[:,:,c][i-1]) / (num+1))
                            if crf_debevec[:,:,c][i+num] == 0 or crf_debevec[:,:,c][i+num] == inf or crf_debevec[:,:,c][i+num] > 100:
                                continue                      
                            break
                    else:
                        continue         
                        
            for i in range(256):
                print(crf_debevec[:,:][i])

        check_bgr()

        crf_b = crf_b.flatten()
        crf_g = crf_g.flatten()
        crf_r = crf_r.flatten()

        # アウトプット
        pyp.plot(crf_b,color = 'tab:blue')
        pyp.plot(crf_g,color = 'tab:green')
        pyp.plot(crf_r,color = 'tab:orange')

        pyp.xlabel("Measured Intensity",{"fontsize":10})
        pyp.ylabel("Calibrated Intensity",{"fontsize":10})

        # (環境によって都度指定)
        file_gragh = "./Python/HDR/Replaceimg_lambda0/OutputGragh/" + condition + ".jpg"
        pyp.savefig(file_gragh)

        pyp.show()

        # (環境によって都度指定)
        file_curve = "./Python/HDR/Replaceimg_lambda0/ResponseCurve/" + condition + ".npy"
        np.save(file_curve,crf_debevec)


    # 画像をHDR画像として統合
    def MergeDebevec(): 

         # (環境によって都度指定)
        file_reloadcurve = "./Python/HDR/Replaceimg_lambda0/ResponseCurve/" + condition + ".npy"
        crf_debevec = np.load(file_reloadcurve)

    # (createMergeDebevec(src,times(vecter of exposure time values for each image),response(256✖︎1)[,dst]))
        merge_debevec = cv2.createMergeDebevec()
        hdr_debevec = merge_debevec.process(img_list,exporsure_times.copy(),crf_debevec)

        # (環境によって都度指定)
        file_HDR = "./Python/HDR/Replaceimg_lambda0/Created_HDR/excerpt/" + condition + ".hdr"
        cv2.imwrite(file_HDR,hdr_debevec)
        cv2.imshow('dst3',hdr_debevec)


    CalibrateDebevec()
    MergeDebevec()

    cv2.waitKey()
    cv2.destroyAllWindows()


except:
    import sys
    print('Error:',sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))
