import copy

def add_pixel_pattern(helper, ori_image):
    image = copy.deepcopy(ori_image)
    poison_patterns = helper.params['poison_patterns']

    if ori_image.size(0) == 3:
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1

    elif ori_image.size(0) == 1:
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1

    return image


def add_pixel_pattern_dba(epoch, helper, ori_image):
    image = copy.deepcopy(ori_image)
    poison_patterns = helper.params['poison_patterns']
    dba_idx = epoch % 4
    poison_pattern = poison_patterns[6 * dba_idx : 6 * (dba_idx + 1)]

    if ori_image.size(0) == 3:
        for i in range(0, len(poison_pattern)):
            pos = poison_pattern[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1

    elif ori_image.size(0) == 1:
        for i in range(0, len(poison_pattern)):
            pos = poison_pattern[i]
            image[0][pos[0]][pos[1]] = 1

    return image