import numpy as np
from .configs.skeletons_cfg import SKELETON
from .configs import give_idxs_from_labels
SKEL_WITH_IDXS = give_idxs_from_labels(SKELETON)


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
                in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - acappella: ACAPELLA
        - mmpose
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='mmpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):



        if layout == 'acappella':
            self.num_node = 68
            self_link = [(i, i) for i in range(self.num_node)]

            #print("new graph")

            # Generate sequential pairs for all 68 nodes
            all_seq = [(i, i + 1) for i in range(67)]

            # Manually define slices based on standard 68-point Dlib/VoxCeleb format
            # These indices ensure we don't accidentally connect distant points (e.g., end of jaw to start of eyebrow)
            face      = all_seq[0:16]           # Jawline (0-16)
            eyebrow1  = all_seq[17:21]          # Right eyebrow (17-21)
            eyebrow2  = all_seq[22:26]          # Left eyebrow (22-26)
            nose      = all_seq[27:30]          # Nose bridge (27-30)
            nostril   = all_seq[31:35]          # Nostrils (31-35)

            # Closed loops for critical features
            eye1      = all_seq[36:41] + [(41, 36)] # Right eye (closed)
            eye2      = all_seq[42:47] + [(47, 42)] # Left eye (closed)
            lips      = all_seq[48:59] + [(59, 48)] # Outer lips (closed)
            teeth     = all_seq[60:67] + [(67, 60)] # Inner lips/teeth (closed)

            # Combine all edges to form the full face skeleton
            self.edge = (self_link + face + eyebrow1 + eyebrow2 +
                        nose + nostril + eye1 + eye2 + lips + teeth)

            # Use Point 30 (nose bridge) as the center node for the graph
            self.center = 30



        elif layout == 'acappella1':
            self.num_node = 68
            self_link = [(i, i) for i in range(self.num_node)]
            all = [(i, i + 1) for i in range(20)]

            face = all[slice(0, 16)]
            eyebrown1 = all[slice(17, 21)]
            eyebrown2 = all[slice(22, 26)]
            nose = all[slice(27, 30)]
            nostril = all[slice(31, 35)]
            eye1 = all[slice(36, 41)]
            eye2 = all[slice(42, 47)]
            lips = all[slice(48, 59)]
            teeth = all[slice(60, 67)]
            self.edge = self_link + lips + teeth + face + eyebrown1 + eyebrown2 + nose + nostril + eye1 + eye2
            self.center = 0
            #print(self.edge)
            # ORIGINAL SOURCECODE
            # https://github.com/1adrianb/face-alignment/blob/master/examples/detect_landmarks_in_image.py
            # pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
            #               'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
            #               'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
            #               'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
            #               'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
            #               'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
            #               'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
            #               'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
            #               'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
            #               }
            # Voxceleb distribution
            # slice(0, 17), 'face contour'
            # slice(17, 22), 'right eyebrow'
            # slice(22, 27), 'left eyebrow'
            # slice(27, 36), 'nose'
            # slice(36, 42), 'right eye'
            # slice(42, 48), 'left eye'
            # slice(48, 69), 'mouth'
        elif layout == 'mmpose':
            self.num_node = 55
            num_edges = 55
            self_link = [(i, i) for i in range(self.num_node)]
            idxs = SKEL_WITH_IDXS['label_ids']
            edge_info = SKEL_WITH_IDXS['skeleton_info']

            all = [(idxs[edge_info[i]['link'][0]], idxs[edge_info[i]['link'][1]]) for i in range(num_edges)]

            body_wo_legs = all[4:19]
            b = 78
            hands = all[25:]
            hands = [(edge[0]-b, edge[1]-b) for edge in hands]
            self.edge = self_link + body_wo_legs + hands
            self.center = 0
        elif layout == 'mmpose_hand':
            self.num_node = 42  # Only hands
            num_edges = 42  # Assuming one edge per node (can be adjusted)

            # Create self-loops
            hand_edges = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index finger
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle finger
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring finger
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky finger
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]

            left_hand_offset=0
            right_hand_offset=21

            # Shift hand keypoints for left and right hands
            left_hand_edges = [(a + left_hand_offset, b + left_hand_offset) for a, b in hand_edges]
            right_hand_edges = [(a + right_hand_offset, b + right_hand_offset) for a, b in hand_edges]

            self.edge = left_hand_edges+ right_hand_edges
            print(self.edge)
            self.center = 0

        elif layout == 'mmpose_full':



            self.num_node = 123  # Since 10 keypoints were already removed



            # Final edges: self-links + valid body edges + face edges
            self.edge = [(0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (9, 9),
            (10, 10),
            (13, 13),
            (14, 14),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (20, 20),
            (21, 21),
            (22, 22),
            (23, 23),
            (24, 24),
            (25, 25),
            (26, 26),
            (27, 27),
            (28, 28),
            (29, 29),
            (30, 30),
            (31, 31),
            (32, 32),
            (33, 33),
            (34, 34),
            (35, 35),
            (36, 36),
            (37, 37),
            (38, 38),
            (39, 39),
            (40, 40),
            (41, 41),
            (42, 42),
            (43, 43),
            (44, 44),
            (45, 45),
            (46, 46),
            (47, 47),
            (48, 48),
            (49, 49),
            (50, 50),
            (51, 51),
            (52, 52),
            (53, 53),
            (54, 54),
            (55, 55),
            (56, 56),
            (57, 57),
            (58, 58),
            (59, 59),
            (60, 60),
            (61, 61),
            (62, 62),
            (63, 63),
            (64, 64),
            (65, 65),
            (66, 66),
            (67, 67),
            (68, 68),
            (69, 69),
            (70, 70),
            (71, 71),
            (72, 72),
            (73, 73),
            (74, 74),
            (75, 75),
            (76, 76),
            (77, 77),
            (78, 78),
            (79, 79),
            (80, 80),
            (81, 81),
            (82, 82),
            (83, 83),
            (84, 84),
            (85, 85),
            (86, 86),
            (87, 87),
            (88, 88),
            (89, 89),
            (90, 90),
            (91, 91),
            (92, 92),
            (93, 93),
            (94, 94),
            (95, 95),
            (96, 96),
            (97, 97),
            (98, 98),
            (99, 99),
            (100, 100),
            (101, 101),
            (102, 102),
            (103, 103),
            (104, 104),
            (105, 105),
            (106, 106),
            (107, 107),
            (108, 108),
            (109, 109),
            (110, 110),
            (111, 111),
            (112, 112),
            (113, 113),
            (114, 114),
            (115, 115),
            (116, 116),
            (117, 117),
            (118, 118),
            (119, 119),
            (120, 120),
            (121, 121),
            (122, 122),
            (11, 12),
            (5, 11),
            (6, 12),
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (1, 2),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (81, 82),
            (82, 83),
            (83, 84),
            (84, 85),
            (81, 86),
            (86, 87),
            (87, 88),
            (88, 89),
            (81, 90),
            (90, 91),
            (91, 92),
            (92, 93),
            (81, 94),
            (94, 95),
            (95, 96),
            (96, 97),
            (81, 98),
            (98, 99),
            (99, 100),
            (100, 101),
            (102, 103),
            (103, 104),
            (104, 105),
            (105, 106),
            (102, 107),
            (107, 108),
            (108, 109),
            (109, 110),
            (102, 111),
            (111, 112),
            (112, 113),
            (113, 114),
            (102, 115),
            (115, 116),
            (116, 117),
            (117, 118),
            (102, 119),
            (119, 120),
            (120, 121),
            (121, 122),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),
            (29, 30),
            (30, 31),
            (31, 32),
            (32, 33),
            (33, 34),
            (34, 35),
            (35, 36),
            (36, 37),
            (37, 38),
            (38, 39),
            (39, 40),
            (40, 41),
            (41, 42),
            (42, 43),
            (43, 44),
            (44, 45),
            (45, 46),
            (46, 47),
            (47, 48),
            (48, 49),
            (49, 50),
            (50, 51),
            (51, 52),
            (52, 53),
            (53, 54),
            (54, 55),
            (55, 56),
            (56, 57),
            (57, 58),
            (58, 59),
            (59, 60),
            (60, 61),
            (61, 62),
            (62, 63),
            (63, 64),
            (64, 65),
            (65, 66),
            (66, 67),
            (67, 68),
            (68, 69),
            (69, 70),
            (70, 71),
            (71, 72),
            (72, 73),
            (73, 74),
            (74, 75),
            (75, 76),
            (76, 77),
            (77, 78),
            (78, 79),
            (79, 80)]
            self.center = 0



            #print(self.edge)

        elif layout == 'mmpose_violin':

            self.num_node= 24
            self.edge= [(0,1), (1,2),(3, 4), (4, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10), (10, 11),
                (3, 12), (12, 13), (13, 14), (14, 15), (3, 16), (16, 17), (17, 18), (18, 19),
                (3, 20), (20, 21), (21, 22), (22, 23)
                ]

            #self.edge= [ (0,1), (1,2)]

            self.center = 0

        else:
            raise ValueError("Do Not Exist This Layout.")
        return self.edge, self.num_node

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
