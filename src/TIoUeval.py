import math
from collections import namedtuple
from copy import deepcopy

import numpy as np


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'AREA_RECALL_CONSTRAINT' : 0.8,
        'AREA_PRECISION_CONSTRAINT' : 0.4,
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,
        'GT_SAMPLE_NAME_2_ID':'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID':'res_img_([0-9]+).txt',
        'CRLF':False # Lines are delimited by Windows CRLF format
    }


def calc_tioueval_metrics(pred_bboxes_dict, gt_bboxes_dict,
                         eval_hparams=None, bbox_format='rect', verbose=False):
    """
    현재는 rect(xmin, ymin, xmax, ymax) 형식의 bounding box만 지원함. 다른 형식(quadrilateral,
    poligon, etc.)의 데이터가 들어오면 외접하는 rect로 변환해서 이용하고 있음.
    """

    def one_to_one_match(row, col):
        cont = 0
        for j in range(len(recallMat[0])):
            if recallMat[row,j] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,j] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False
        cont = 0
        for i in range(len(recallMat)):
            if recallMat[i,col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[i,col] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False

        if recallMat[row,col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,col] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
            return True
        return False

    def one_to_one_match_v2(row, col): # here
        if row_sum[row] != 1:
            return False

        if col_sum[col] != 1:
            return False

        if recallMat[row, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and \
                precisionMat[row, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
            return True
        return False

    def num_overlaps_gt(gtNum): # gt를 주면 det한 수 -> one to many
        cont = 0
        for detNum in range(len(detRects)):
            if recallMat[gtNum,detNum] > 0 :
                cont = cont +1
        return cont

    def num_overlaps_det(detNum): # det를 주면 걸린 gt 수 -> many to one
        cont = 0
        for gtNum in range(len(recallMat)):
            if recallMat[gtNum,detNum] > 0 :
                cont = cont +1
        return cont

    def is_single_overlap(row, col): # one to one
        if num_overlaps_gt(row)==1 and num_overlaps_det(col)==1:
            return True
        else:
            return False

    def one_to_many_match(gtNum): # 잘게 자른 경우 검사 -> precision 먼저 검사하고 모아서 recall 검사
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0: # 미리 gtNum을 검사하면 쓸모없는 반복 제거 가능
                if precisionMat[gtNum,detNum] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                    many_sum += recallMat[gtNum,detNum]
                    detRects.append(detNum)
        if round(many_sum,4) >=eval_hparams['AREA_RECALL_CONSTRAINT'] :
            return True,detRects
        else:
            return False,[]

    def many_to_one_match(detNum): # 하나로 뭉친 경우 검사 -> recall 먼저 검사하고 모아서 precision 검사
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                if recallMat[gtNum,detNum] >= eval_hparams['AREA_RECALL_CONSTRAINT'] :
                    many_sum += precisionMat[gtNum,detNum]
                    gtRects.append(gtNum)
        if round(many_sum,4) >=eval_hparams['AREA_PRECISION_CONSTRAINT'] :
            return True,gtRects
        else:
            return False,[]

    def area(a, b): # 겹치는 영역
            dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
            dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
            if (dx>=0) and (dy>=0):
                    return dx*dy
            else:
                    return 0.

    def center(r):
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
        return Point(x,y)

    def point_distance(r1, r2):
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty )


    def center_distance(r1, r2): # 박스를 입력 받음
        return point_distance(center(r1), center(r2))

    def diag(r): # 대각선?
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)

    if eval_hparams is None:
        eval_hparams = default_evaluation_params()

    if bbox_format != 'rect':
        raise NotImplementedError

    # bbox들이 rect 이외의 형식으로 되어있는 경우 rect 형식으로 변환
    _pred_bboxes_dict, _gt_bboxes_dict= deepcopy(pred_bboxes_dict), deepcopy(gt_bboxes_dict)
    pred_bboxes_dict, gt_bboxes_dict = dict(), dict()
    for sample_name, bboxes in _pred_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            pred_bboxes_dict = _pred_bboxes_dict
            break

        pred_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            pred_bboxes_dict[sample_name].append(rect)
    for sample_name, bboxes in _gt_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            gt_bboxes_dict = _gt_bboxes_dict
            break

        gt_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            gt_bboxes_dict[sample_name].append(rect)

    perSampleMetrics = {} #

    methodRecallSum = 0 #
    methodPrecisionSum = 0 #

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    numGt = 0
    numDet = 0

    for sample_name in gt_bboxes_dict:

        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0. # 매칭되면 1 or 0.8 더해주는 건가?
        precisionAccum = 0.
        gtRects = [] # Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax') 저장
        detRects = []
        gtPolPoints = [] # [np.array([xmin, ymin, xmax, ymax]),,,]
        detPolPoints = []
        pairs = []
        evaluationLog = ""

        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])

        pointsList = gt_bboxes_dict[sample_name]

        for n in range(len(pointsList)):
            points = pointsList[n] # [xmin, ymin, xmax, ymax]
            gtRect = Rectangle(*points)
            gtRects.append(gtRect)
            gtPolPoints.append(np.array(points).tolist())

        evaluationLog += "GT rectangles: " + str(len(gtRects)) + '\n'

        if sample_name in pred_bboxes_dict:
            pointsList = pred_bboxes_dict[sample_name]

            for n in range(len(pointsList)):
                points = pointsList[n]
                detRect = Rectangle(*points)
                detRects.append(detRect)
                detPolPoints.append(np.array(points).tolist())

            evaluationLog += "DET rectangles: " + str(len(detRects)) + '\n'

            if len(gtRects)==0:
                recall = 1
                precision = 0 if len(detRects)>0 else 1

            if len(detRects)>0:
                #Calculate recall and precision matrixs
                outputShape=[len(gtRects),len(detRects)] # 매트릭스 shape
                recallMat = np.empty(outputShape)
                precisionMat = np.empty(outputShape) # here iou mat을 만들어야겠다
                iouMat = np.empty(outputShape) # add 1
                TiouMat = np.empty(outputShape) # add 2
                gtRectMat = np.zeros(len(gtRects),np.int8) # row 통과 여부
                detRectMat = np.zeros(len(detRects),np.int8) # col 통과 여부
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)): # here 매트릭스 채우기
                        rG = gtRects[gtNum]
                        rD = detRects[detNum]
                        intersected_area = area(rG,rD)
                        rgDimensions = ( (rG.xmax - rG.xmin+1) * (rG.ymax - rG.ymin+1) )
                        rdDimensions = ( (rD.xmax - rD.xmin+1) * (rD.ymax - rD.ymin+1))
                        recallMat[gtNum,detNum] = 0 if rgDimensions==0 else  intersected_area / rgDimensions # recall area 매트릭스
                        precisionMat[gtNum,detNum] = 0 if rdDimensions==0 else intersected_area / rdDimensions # precision area 매트릭스
                        iouMat[gtNum, detNum] = 0 if (rgDimensions + rdDimensions - intersected_area)==0 else intersected_area / (rgDimensions + rdDimensions - intersected_area) # add 3
                        tmp1 = 2 * iouMat[gtNum, detNum] * iouMat[gtNum, detNum] * recallMat[gtNum,detNum] * precisionMat[gtNum,detNum] # add 4
                        tmp2 = iouMat[gtNum, detNum] * (recallMat[gtNum,detNum] + precisionMat[gtNum,detNum]) # add 5
                        TiouMat[gtNum, detNum] = 0 if tmp2==0 else tmp1 / tmp2 # add 6

                recall_cond = recallMat >= eval_hparams['AREA_RECALL_CONSTRAINT'] # 리콜 조건 충족 검사, 단순 검사 one to one 만 가능
                precision_cond = precisionMat >= eval_hparams['AREA_PRECISION_CONSTRAINT'] # precision 조건 충족 검사, 단순 검사 one to one 만 가능
                cond = recall_cond & precision_cond # 둘다 통과
                col_sum = np.sum(cond, axis=0) # pred 하나에 몇개의 gt가 통과
                row_sum = np.sum(cond, axis=1) # gt 하나에 몇개가 통과

                # Find one-to-one matches
                evaluationLog += "Find one-to-one matches\n"
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                            match = one_to_one_match_v2(gtNum, detNum) # threshold를 통과한게 서로 하나뿐
                            if match is True :
                                #in deteval we have to make other validation before mark as one-to-one
                                if is_single_overlap(gtNum, detNum) is True : # 아에 겹치는게 없어야 함
                                    rG = gtRects[gtNum]
                                    rD = detRects[detNum]
                                    normDist = center_distance(rG, rD)
                                    normDist /= diag(rG) + diag(rD)
                                    normDist *= 2.0 # 두 사각형의 크기에 따른 거리 (박스 크기가 크면 쪼금 더 멀어도 괜찮다)
                                    if normDist < eval_hparams['EV_PARAM_IND_CENTER_DIFF_THR'] : # 박스의 거리가 대각선의 합보다 멀면 안댐
                                        gtRectMat[gtNum] = 1
                                        detRectMat[detNum] = 1
                                        recallAccum += TiouMat[gtNum, detNum] # add 6
                                        precisionAccum += TiouMat[gtNum, detNum] # add 7
                                        pairs.append({'gt':gtNum,'det':detNum,'type':'OO'})
                                        evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                    else:
                                        evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " normDist: " + str(normDist) + " \n"
                                else:
                                    evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " not single overlap\n"
                # Find one-to-many matches
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtRects)):
                    match,matchesDet = one_to_many_match(gtNum) # 검사 여부 확인하고 True False, many에 검출된 번호 리스트
                    if match is True :
                        evaluationLog += "num_overlaps_gt=" + str(num_overlaps_gt(gtNum))
                        #in deteval we have to make other validation before mark as one-to-one
                        if num_overlaps_gt(gtNum)>=2 :# 여러개 겹친거 맞는지 확인
                            gtRectMat[gtNum] = 1    # 다른 박스랑 어쩌다 겹친건 봐줌 하지만 대놓고 겹치면 감점


                            if len(matchesDet)==1:  # add 8
                                recallAccum += TiouMat[gtNum, matchesDet[0]]
                                precisionAccum += TiouMat[gtNum, matchesDet[0]]
                            else:   # add 9
                                iou = np.sum(iouMat[gtNum, matchesDet])
                                tmp_recall = np.sum(recallMat[gtNum, matchesDet])
                                tmp_precision = np.mean(precisionMat[gtNum, matchesDet])
                                Tiou = (2 * iou * iou * tmp_recall * tmp_precision) / (iou * (tmp_recall + tmp_precision))
                                recallAccum += Tiou
                                precisionAccum += (Tiou * len(matchesDet))
                            
                            # recallAccum += (TiouMat[gtNum, matchesDet[0]] if len(matchesDet)==1 else eval_hparams['MTYPE_OM_O'])
                            # precisionAccum += (TiouMat[gtNum, matchesDet[0]] if len(matchesDet)==1 else eval_hparams['MTYPE_OM_O']*len(matchesDet))

                            pairs.append({'gt':gtNum,'det':matchesDet,'type': 'OO' if len(matchesDet)==1 else 'OM'})
                            for detNum in matchesDet :
                                detRectMat[detNum] = 1
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(matchesDet) + " not single overlap\n"

                # Find many-to-one matches
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detRects)):
                    match,matchesGt = many_to_one_match(detNum)
                    if match is True :
                        #in deteval we have to make other validation before mark as one-to-one
                        if num_overlaps_det(detNum)>=2 :
                            detRectMat[detNum] = 1


                            if len(matchesGt)==1:  # add 10
                                recallAccum += TiouMat[matchesGt[0], detNum]
                                precisionAccum += TiouMat[matchesGt[0], detNum]
                            else:   # add 11
                                iou = np.sum(iouMat[matchesGt, detNum])
                                tmp_recall = np.mean(recallMat[matchesGt, detNum])
                                tmp_precision = np.sum(precisionMat[matchesGt, detNum])
                                Tiou = (2 * iou * iou * tmp_recall * tmp_precision) / (iou * (tmp_recall + tmp_precision))
                                recallAccum += (Tiou * len(matchesGt))
                                precisionAccum += Tiou

                            # recallAccum += (TiouMat[matchesGt[0], detNum] if len(matchesGt)==1 else eval_hparams['MTYPE_OM_M']*len(matchesGt))
                            # precisionAccum += (TiouMat[matchesGt[0], detNum] if len(matchesGt)==1 else eval_hparams['MTYPE_OM_M'])

                            pairs.append({'gt':matchesGt,'det':detNum,'type': 'OO' if len(matchesGt)==1 else 'MO'})
                            for gtNum in matchesGt :
                                gtRectMat[gtNum] = 1
                            evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(matchesGt) + " with Det #" + str(detNum) + " not single overlap\n"

                numGtCare = len(gtRects)
                if numGtCare == 0:
                    recall = float(1)
                    precision = float(0) if len(detRects)>0 else float(1)
                else:
                    recall = float(recallAccum) / numGtCare
                    precision =  float(0) if len(detRects)==0 else float(precisionAccum) / len(detRects)
                hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)

        methodRecallSum += recallAccum
        methodPrecisionSum += precisionAccum
        numGt += len(gtRects)
        numDet += len(detRects)

        perSampleMetrics[sample_name] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'recall_matrix': [] if len(detRects)>100 else recallMat.tolist(),
            'precision_matrix': [] if len(detRects)>100 else precisionMat.tolist(),
            'gt_bboxes': gtPolPoints,
            'det_bboxes': detPolPoints,
        }

        if verbose:
            perSampleMetrics[sample_name].update(evaluation_log=evaluationLog)

    methodRecall = 0 if numGt==0 else methodRecallSum/numGt
    methodPrecision = 0 if numDet==0 else methodPrecisionSum/numDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall,'hmean': methodHmean}

    resDict = {'calculated': True, 'Message': '', 'total': methodMetrics,
               'per_sample': perSampleMetrics, 'eval_hparams': eval_hparams}

    return resDict
