class DataLoader:
    
    def __init__(self,graph,node_feats=[],edge_index=None,label={}):
        self.graph = graph