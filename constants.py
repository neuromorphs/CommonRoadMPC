

#track waypointsway
# 

oval_points_x = [546, 566, 586, 606, 626, 646, 666, 686, 706, 726, 746, 766, 785, 804, 822, 838, 852, 866,
                876, 885, 891, 897, 902, 904, 905, 906, 907, 906, 904, 899, 893, 888, 879, 870, 856, 842,
                826, 807, 787, 768, 749, 729, 709, 689, 669, 649, 629, 609, 589, 569, 549, 529, 509, 489,
                469, 449, 429, 409, 389, 369, 349, 329, 309, 289, 269, 249, 230, 212, 195, 179, 164, 153,
                143, 135, 128, 124, 121, 119, 118, 119, 119, 120, 124, 128, 135, 142, 151, 160, 173, 188,
                202, 220, 239, 257, 276, 296, 316, 336, 356, 376, 396, 416, 436, 456, 476, 496, 516, 536]
oval_points_y = [156, 156, 156, 155, 155, 155, 155, 153, 155, 156, 157, 162, 172, 185, 199, 217, 235, 255,
            275, 295, 315, 335, 355, 375, 395, 415, 435, 455, 475, 495, 515, 535, 554, 574, 594, 613,
            631, 649, 661, 671, 678, 678, 678, 678, 679, 679, 679, 679, 681, 681, 680, 679, 679, 679,
            679, 679, 679, 678, 678, 678, 678, 678, 677, 677, 674, 667, 656, 642, 627, 608, 589, 570,
            550, 530, 511, 491, 471, 451, 431, 411, 391, 371, 351, 331, 311, 291, 272, 253, 236, 217,
            199, 185, 172, 163, 157, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156]

track_2_points_x = [419, 439, 459, 479, 499, 519, 539, 559, 579, 599, 619, 638, 649, 658, 662, 665, 666, 669, 
                    670, 672, 674, 675, 689, 709, 728, 746, 765, 784, 802, 821, 841, 859, 878, 894, 903, 906, 
                    906, 904, 896, 885, 868, 848, 828, 808, 788, 769, 750, 731, 711, 692, 673, 653, 633, 614, 
                    595, 576, 557, 539, 520, 500, 482, 464, 445, 426, 406, 387, 371, 367, 358, 343, 325, 305, 
                    287, 268, 249, 230, 212, 193, 174, 154, 136, 125, 119, 118, 118, 119, 119, 119, 119, 119, 
                    119, 123, 132, 145, 165, 185, 205, 224, 236, 242, 242, 242, 242, 242, 241, 241, 243, 248, 
                    259, 275, 295, 315, 335, 355, 375, 395, 415]
track_2_points_y = [136, 136, 137, 140, 141, 142, 144, 145, 146, 147, 156, 172, 191, 211, 231, 251, 271, 291, 
                    311, 331, 351, 371, 389, 402, 412, 423, 434, 447, 458, 470, 482, 494, 505, 523, 543, 563, 
                    583, 603, 622, 640, 655, 653, 647, 638, 627, 616, 606, 596, 586, 576, 566, 556, 544, 532, 
                    519, 508, 494, 482, 481, 488, 500, 512, 523, 529, 523, 514, 526, 546, 565, 581, 593, 603, 
                    614, 625, 636, 647, 660, 673, 681, 676, 661, 642, 622, 602, 582, 562, 542, 522, 502, 482, 
                    462, 442, 422, 404, 394, 387, 381, 372, 352, 332, 312, 292, 272, 252, 232, 212, 192, 172, 
                    152, 137, 134, 132, 135, 135, 135, 135, 136]


#no action
u_0 = [
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
]

#steer hard to left
u_1 = [
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
    [0.4, 0],
]

#steer hard to right
u_2 = [
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
    [-0.4, 0],
]

#steer softly to left
u_3 = [
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
    [0.2, 0],
]

#steer softly to left
u_4 = [
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
    [-0.2, 0],
]


#break
u_5 = [
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
    [0.0, -3],
]

#break and left
u_6 = [
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
    [0.4, -3],
]

#break and right
u_7 = [
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
    [-0.4, -3],
]


#accelerate
u_8 = [
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
]

#accelerate and left
u_8 = [
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
    [0.4, 1],
]

#accelerate and right
u_9 = [
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
    [-0.4, 1],
]

u_dist = [u_0, u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9]