import torch
import ultralytics


def cal_intersection(box):
    """
    Bounding_Box 간의 교차 영역을 계산하여 겹치는 부분의 비율을 더 작은 면적을 기준으로 계산하여 반환합니다.\n
    겹치는 영역을 최소 면적으로 나눈 후, 대각선 요소를 제거하여 반환합니다.\n
    __________________________________________________________________________

    Args:
        box (torch.Tensor): 각 Bounding_Box의 좌상단(x1, y1)과 우하단(x2, y2) 좌표를 포함하는 Tensor. 모양은 [N, 4]이어야 합니다, 여기서 N은 박스의 개수입니다.
    __________________________________________________________________________

    Returns:
        torch.Tensor: 각 Bounding_Box 간의 겹치는 영역의 비율을 나타내는 Tensor. 모양은 [N, N]이며 대각선 값은 0입니다.
    """
    box=box.clone()
    box_tile0 = box[:,0].tile((box.shape[0],1))
    top = torch.max(torch.concat([box_tile0[None,:],box_tile0.t()[None,:]],axis=0),axis=0)[0]
    box_tile2 = box[:,2].tile((box.shape[0],1))
    bottom = torch.min(torch.concat([box_tile2[None,:],box_tile2.t()[None,:]],axis=0),axis=0)[0]
    w = bottom-top
    w = torch.where(w>0,w,0)
    
    box_tile1 = box[:,1].tile((box.shape[0],1))
    left = torch.max(torch.concat([box_tile1[None,:],box_tile1.t()[None,:]],axis=0),axis=0)[0]
    box_tile3 = box[:,3].tile((box.shape[0],1))
    right = torch.min(torch.concat([box_tile3[None,:],box_tile3.t()[None,:]],axis=0),axis=0)[0]
    h = right-left
    h = torch.where(h>0,h,0)
    
    area_tile=(box[:,2]-box[:,0])*(box[:,3]-box[:,1]).tile((box.shape[0],1))
    min_area = torch.min(torch.concat([area_tile[None,:],area_tile.t()[None,:]],axis=0),axis=0)[0]
    
    return  w*h/min_area - torch.eye(box.shape[0])
    
    
def merge_overlapping_boxes(_box,thrsh=0.9):
    """
    같은 클래스의 Bounding_Box들 중 서로 겹치는 Bounding_Box를 하나로 병합하여 중복을 제거합니다. 겹치는 부분이 주어진 임계값(thrsh)보다 큰 경우, Bounding_Box를 병합합니다.\n
    삭제된 Bounding_Box를 제외한 최종 Bounding_Box 목록을 반환합니다.\n
    __________________________________________________________________________
    
    Args:
        _box (torch.Tensor): 각 Bounding_Box의 좌상단(x1, y1)과 우하단(x2, y2) 좌표를 포함하는 Tensor. 모양은 [N, 4]이어야 합니다, 여기서 N은 Bounding_Box의 개수입니다.
        thrsh (float, optional): 두 Bounding_Box가 병합되어야 하는 겹치는 영역의 최소 비율. 기본값은 0.9입니다.
    __________________________________________________________________________
    
    Returns:
        torch.Tensor: 중복이 제거된 Bounding_Box의 좌표를 포함하는 Tensor. 모양은 [M, 4]이며, M은 중복이 제거된 Bounding_Box의 개수입니다.
    """
    box=_box.clone()
    intersec = cal_intersection(box)
    mask=(torch.triu(intersec)>thrsh).nonzero(as_tuple=True)
    if len(mask[0])==0: return box
    overlapping_boxes=torch.concat([box[m][None,:] for m in torch.stack(mask,axis=1)])
    overlapping_boxes[:,:,[2,3]]*=-1
    overlaped_areas=torch.max(overlapping_boxes,axis=1)[0]
    overlaped_areas[:,[2,3]]*=-1
    
    box[mask[0]]=overlaped_areas
    box[mask[1]]=-1
    
    return box[box[:,-1]!=-1]


def process_part_bee(_box, thrsh=0.7):
    """
    part_bee 내부에 위치한(thrsh값 보다 큰) head와 abdomen의 Bounding_Box를 처리합니다. part_bee와 head 또는 abdomen가 겹치는 비율이 주어진 임계값(thrsh)보다 높다면 해당 head 또는 abdomen의 Bounding_Box를 삭제합니다.\n
    삭제된 Bounding_Box를 제외한 최종 Bounding_Box 목록을 반환합니다.\n
    __________________________________________________________________________
    
    Args:
        _box (torch.Tensor): 각 Bounding_Box의 좌상단(x1, y1)과 우하단(x2, y2) 좌표와 part_bee를 나타내는 label을 포함하는 Tensor. 모양은 [N, 5]이어야 합니다, 여기서 N은 Bounding_Box의 개수입니다.
        thrsh (float, optional): head와 abdomen이 삭제되어야 하는 겹쳐진 정도의 최소 비율. 기본값은 0.7입니다.
    __________________________________________________________________________
    Returns:
        torch.Tensor: part_bee와 겹친 head와 abdomen이 제거된 Bounding_Box의 좌표를 포함하는 Tensor. 모양은 [M, 5]이며, M은 남은 Bounding_Box의 개수입니다.
    """
    box = _box.clone() 
    intersec = cal_intersection(box) 
    # 마스크 생성
    mask_part_bee = box[:, -1] == 4    
    mask_head_abdomen = (box[:, -1] == 3) | (box[:, -1] == 0) 
    # 겹치는 비율 계산해서 선택하고 삭제 마스크
    overlapping = intersec > thrsh  
    overlapping_with_part_bee = overlapping[mask_part_bee][:, mask_head_abdomen]
    deletion_mask = overlapping_with_part_bee.any(dim=0) 
    
    # 삭제 마스크를 head_abdomen에 적용
    full_deletion_mask = torch.zeros_like(mask_head_abdomen, dtype=torch.bool)
    full_deletion_mask[mask_head_abdomen] = deletion_mask

    box[full_deletion_mask, -1] = -1  # 삭제할 상자 마킹
    return box[box[:, -1] != -1]  # 삭제 안 된 애들 반환


def process_flyingbee(_box, thrsh=0.7):
    """
    flyingbee 내부에 위치한(thrsh값 보다 큰) head와 abdomen의 Bounding_Box를 처리합니다. flyingbee와 head 또는 abdomen가 겹치는 비율이 주어진 임계값(thrsh)보다 높다면 해당 head 또는 abdomen의 Bounding_Box를 삭제합니다.\n
    삭제된 Bounding_Box를 제외한 최종 Bounding_Box 목록을 반환합니다.\n
    __________________________________________________________________________
    
    Args:
        _box (torch.Tensor): 각 Bounding_Box의 좌상단(x1, y1)과 우하단(x2, y2) 좌표와 flyingbee를 나타내는 label을 포함하는 Tensor. 모양은 [N, 5]이어야 합니다, 여기서 N은 Bounding_Box의 개수입니다.
        thrsh (float, optional): head와 abdomen이 삭제되어야 하는 겹쳐진 정도의 최소 비율. 기본값은 0.7입니다.
    __________________________________________________________________________
    
    Returns:
        torch.Tensor:  flyingbee와 겹친 head와 abdomen이 제거된 Bounding_Box의 좌표를 포함하는 Tensor. 모양은 [M, 5]이며, M은 남은 Bounding_Box의 개수입니다.
    """
    box = _box.clone() 
    intersec = cal_intersection(box)
    # 마스크 생성
    mask_flying_bee = box[:, -1] == 2
    mask_head_abdomen = (box[:, -1] == 3) | (box[:, -1] == 0)
    # 겹치는 비율 계산해서 선택하고 삭제 마스크
    overlapping = intersec > thrsh
    overlapping_with_flying_bee = overlapping[mask_flying_bee][:, mask_head_abdomen]
    deletion_mask = overlapping_with_flying_bee.any(dim=0)

    # 삭제 마스크를 head_abdomen에 적용
    full_deletion_mask = torch.zeros_like(mask_head_abdomen, dtype=torch.bool)
    full_deletion_mask[mask_head_abdomen] = deletion_mask

    box[full_deletion_mask, -1] = -1  # 삭제할 상자 마킹
    return box[box[:, -1] != -1] # 삭제 안 된 애들 반환


def centerline_association_method(box_bee, box_head_abdomen, thrsh=0.7):
    """
    bee와 head 및 abdomen의 Bounding_Box들을 "Centerline Association Method"를 이용하여 연관시키는 메서드입니다. Overlapping 비율을 기준으로 연관된 Bounding_Box 순서쌍들을 찾아 해당 Bounding_Box들을 반환합니다.
    __________________________________________________________________________\n
    
    "Centerline Association Method":\n
        \t1. "bee"의 BoundingBox 중심점 계산: "bee"의 BoundingBox의 중심점을 계산합니다. 중심점은 BoundingBox의 네 꼭지점을 기반으로 하며, 평균 x 좌표와 평균 y좌표를 이용하여 계산합니다. \n
        \t2. "head"와 "abdomen"의 BoundingBox 중심점 계산: "bee"의 BoundingBox 내부(thrsh > 0.7)에 있는 각 "head"와 "abdomen"의 BoundingBox의 중심점을 계산합니다. 중심점은 BoundingBox의 네 꼭지점을 기반으로 하며, 평균 x 좌표와 평균 y 좌표를 사용하여 구합니다.\n
        \t3. 각 "head"와 "abdomen"의 중심점을 이은 선을 계산: 각 중심선을 이은 선을 수학적으로 계산합니다. 즉, 각 중심점을 이은 선의 방정식 계산합니다.\n
        \t4. "bee"의 중심점과 가장 가까운 선을 계산: 계산된 선과 "bee"의 중심점 사이의 수직 거리가 가장 작은 선을 찾습니다.\n
        \t5. 해당 선을 이루는 "head"와 "abdomen" 순서쌍이 해당 "bee"에 속하는 애들이라고 판단합니다.\n
    __________________________________________________________________________
    
    Args:
        box_bee (torch.Tensor): bee를 나타내는 Bounding_Box의 좌상단(x1, y1)과 우하단(x2, y2) 좌표를 포함하는 Tensor. 모양은 [N, 4]이어야 하며, N은 Bounding_Box의 개수입니다.
        box_head_abdomen (torch.Tensor): head와 abdomen을 나타내는 Bounding_Box의 좌상단(x1, y1)과 우하단(x2, y2) 좌표를 포함하는 Tensor. 모양은 [M, 4]이며, M은 Bounding_Box의 개수입니다.
        thrsh (float, optional): 중심선과의 거리를 계산할 때 사용되는 겹쳐진 정도의 최소 비율. 기본값은 0.7입니다.
    __________________________________________________________________________
    
    Returns:
        tuple: 연관된 bee와 head, abdomen의 Bounding_Box 내용을 포함하는 tuple. 각 Tensor의 모양은 원래 입력과 동일합니다.
    """
    # 중심점 계산
    bee_centers = (box_bee[:, :2] + box_bee[:, 2:4]) / 2
    head_abdomen_centers = (box_head_abdomen[:, :2] + box_head_abdomen[:, 2:4]) / 2

    # 겹치는 비율 > 0.7을 만족하는 상자 계산
    combined_boxes = torch.cat((box_bee, box_head_abdomen), dim=0)
    intersec = cal_intersection(combined_boxes)
    overlapping = intersec > thrsh
    mask_overlapping = overlapping[:len(box_bee), len(box_bee):]

    # 중심점을 이은 선의 방정식 계산
    bee_centers_expanded = bee_centers[:, None, :]
    distances = torch.abs((head_abdomen_centers[:, 1] - bee_centers_expanded[:, :, 1]) * bee_centers_expanded[:, :, 0] -
                          (head_abdomen_centers[:, 0] - bee_centers_expanded[:, :, 0]) * bee_centers_expanded[:, :, 1] +
                          head_abdomen_centers[:, 0] * bee_centers_expanded[:, :, 1] - head_abdomen_centers[:, 1] * bee_centers_expanded[:, :, 0]) / \
               torch.sqrt((head_abdomen_centers[:, 1] - bee_centers_expanded[:, :, 1]) ** 2 + (head_abdomen_centers[:, 0] - bee_centers_expanded[:, :, 0]) ** 2)

    # 중심점과 가장 가까운 선을 계산
    distances[~mask_overlapping] = float('inf')  # 겹치지 않는 부분은 무한대로 설정
    if distances.dim()==1: distances=distances[None,:]
    min_distances_idx = torch.argmin(distances, dim=1)

    # 해당 순서쌍에 해당하는 박스들 반환
    associated_boxes = box_bee, box_head_abdomen[min_distances_idx]
    return associated_boxes


def process_bee(_box, thrsh=0.7):
    """
    bee 내부에 위치한(thrsh값 보다 큰) head/abdomen가 내부에 혼자 있다면 해당 Bounding_Box를 삭제 처리하고, "Centerline Association Method"를 사용하여 특정 기준에 따라 Bounding_Box를 삭제하는 함수입니다.\n
    최종적으로 삭제된 Bounding_Box를 제외한 Bounding_Box 목록을 반환합니다.\n
    __________________________________________________________________________\n
        
    "Centerline Association Method":\n
        \t1. "bee"의 BoundingBox 중심점 계산: "bee"의 BoundingBox의 중심점을 계산합니다. 중심점은 BoundingBox의 네 꼭지점을 기반으로 하며, 평균 x 좌표와 평균 y좌표를 이용하여 계산합니다. \n
        \t2. "head"와 "abdomen"의 BoundingBox 중심점 계산: "bee"의 BoundingBox 내부(thrsh > 0.7)에 있는 각 "head"와 "abdomen"의 BoundingBox의 중심점을 계산합니다. 중심점은 BoundingBox의 네 꼭지점을 기반으로 하며, 평균 x 좌표와 평균 y 좌표를 사용하여 구합니다.\n
        \t3. 각 "head"와 "abdomen"의 중심점을 이은 선을 계산: 각 중심선을 이은 선을 수학적으로 계산합니다. 즉, 각 중심점을 이은 선의 방정식 계산합니다.\n
        \t4. "bee"의 중심점과 가장 가까운 선을 계산: 계산된 선과 "bee"의 중심점 사이의 수직 거리가 가장 작은 선을 찾습니다.\n
        \t5. 해당 선을 이루는 "head"와 "abdomen" 순서쌍이 해당 "bee"에 속하는 애들이라고 판단합니다.\n
    __________________________________________________________________________
    
    Args:
        _box (torch.Tensor): bee과 head/abdomen을 포함하는 Bounding_Box의 좌상단(x1, y1)과 우하단(x2, y2) 좌표 및 분류 label을 포함하는 Tensor. 모양은 [N, 5]이며, N은 Bounding_Box의 개수입니다.
        thrsh (float, optional): bee와 head/abdomen이 겹쳐진 정도의 최소 비율을 결정하는 임계값. 기본값은 0.7입니다.
    __________________________________________________________________________
    
    Returns:
        torch.Tensor: 처리된 Bounding_Box가 담긴 Tensor. 입력 Bounding_Box와 동일한 형태이지만, 특정 조건에 따라 삭제된 Bounding_Box가 제외됩니다.
    """
    box = _box.clone() 
    mask_bee = box[:, -1] == 1
    mask_head_abdomen = (box[:, -1] == 3) | (box[:, -1] == 0)
    intersec = cal_intersection(box)
    overlapping = intersec > thrsh

    # "Centerline Association Method"로 순성쌍 찾기 (? 사용안됨, 오류남)
    # centerline_associated_boxes = centerline_association_method(box[mask_bee], box[mask_head_abdomen])

    # Bee와 Head/Abdomen 간의 겹치는 정도 행렬 계산 (70% 이상 겹치는 부분만 고려)
    overlapping_with_head_abdomen = (overlapping[mask_bee][:, mask_head_abdomen] > thrsh).float()
    single_head_abdomen_mask = overlapping_with_head_abdomen.sum(dim=1) == 1 # 혼자 있는 애들 뽑아 (70% 이상 겹치는 부분이 하나만 있는 경우)
    deletion_mask_for_head_abdomen = overlapping_with_head_abdomen[single_head_abdomen_mask].any(dim=0) # 삭제 마스크
    box[mask_head_abdomen][deletion_mask_for_head_abdomen, -1] = -1  # 해당 박스 삭제

    # "Centerline Association Method"에 따른 삭제 마스크 계산
    deletion_mask_for_bee = overlapping[mask_bee][:, mask_head_abdomen].any(dim=1)
    
    box[mask_bee][deletion_mask_for_bee, -1] = -1 # 삭제할 상자 마킹
    return box[box[:, -1] != -1]  # 삭제 안 된 애들 반환


def HHB_boxpostprocess(box, kargs):
    """
    Bounding_Box의 후처리를 수행하여, 중복되는 Bounding_Box를 병합하고 모든 클래스가 bee로 수렴하도록 하는 함수입니다. 이 함수는 벌의 Bounding_Box를 분석하고 가공하여 최종 결과를 반환합니다.\n
    Bounding_Box의 format(box)를 확인하고 (x1,y1,x2,y2,scr,cls)로 맞춥니다.\n
    kargs에서 partial_threshold(p_thrsh)와 overlap_threshold(o_thrsh)을 설정합니다.\n
    순서대로 merge_overlapping_boxes, part_bee와 flyingbee 후처리 함수를 시행하고 part_bee와 flyingbee를 bee 클래스로 변환합니다.\n
    bee 후처리 함수 시행 후 모든 클래스가 bee로 수렴 하도록 처리해줍니다.\n
    최종적으로 bee의 Bounding_Box를 반환합니다.\n
    __________________________________________________________________________
    
    Args:
        box (torch.Tensor): 입력 Bounding_Box를 포함하는 Tensor. 모양은 [N, 5]이며, 각 Bounding_Box는 (x1, y1, x2, y2, score, class) 또는 (x, y, w, h, score, class) 형식을 가집니다.
        kargs (dict): 함수의 동작을 제어하는 매개변수를 포함하는 dictionary. 키 'Bpop.p_thrsh'는 partial_threshold값(p_thrsh)을 제어하고, 'Bpop.0_thrsh'는 overlap_threshold값(o_thrsh)을 제어합니다.
    __________________________________________________________________________
    
    Returns:
        torch.Tensor: 처리된 Bounding_Box가 포함된 Tensor. 입력 텐서와 동일한 형식이지만, 특정 기준에 따라 처리된 Bounding_Box로 업데이트됩니다.
    """
    
    if isinstance(box,(tuple,list)):
        box=box[0].boxes.data
    box=box.clone().detach().cpu()
    if len(box)==0: return box
    
    # 첫 번째 bounding box의 w와 h를 체크
    w, h = box[0, 2], box[0, 3]
    if w < 1 and h < 1: # w와 h가 1보다 작은 값인 경우 (x, y, w, h)로 간주
        box = ultralytics.utils.ops.xywhn2xyxy(box, w=640, h=640, padw=0, padh=0)
        
    p_thrsh=kargs.get('Bpop.p_thrsh',0.9)
    o_thrsh=kargs.get('Bpop.0_thrsh',0.7)

    # merge_overlapping_boxes 시행하고 하나로 합침 -> 이제 merged_box가 최신 업데이트
    merged_box = torch.cat([merge_overlapping_boxes(box[box[:, -1] == cls], o_thrsh) for cls in [1, 2, 4, 0, 3]], axis=0)
    
    # part_bee와 flyingbee 처리
    merged_box = process_part_bee(merged_box, o_thrsh)
    merged_box = process_flyingbee(merged_box, o_thrsh)

    # part_bee와 flyingbee를 bee 클래스로 변환
    mask_part_bee = merged_box[:, -1] == 4
    mask_flying_bee = merged_box[:, -1] == 2
    merged_box[mask_part_bee, -1] = 1
    merged_box[mask_flying_bee, -1] = 1

    # bee 처리
    merged_box = process_bee(merged_box, o_thrsh)
    
    # 겹치는 영역 계산
    intersec = cal_intersection(merged_box)

    mask1 = torch.logical_or(merged_box[:, -1] == 0, merged_box[:, -1] == 3)
    mask2 = torch.logical_and(intersec > p_thrsh, mask1.tile((mask1.shape[0], 1)))

    merged_box[~mask1, -2] *= 0.6 + 0.2 * mask2[~mask1].sum(axis=1)
    merged_box[~mask1, -1] = 1

    merged_box[mask1, -2] *= 0.5
    merged_box[mask1, -1] = 1

    merged_box[mask2.any(axis=0)] = -1
    return merged_box[merged_box[:, 0] != -1]