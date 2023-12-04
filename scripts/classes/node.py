class Node:
    def __init__(self,name:str,cell:str,output:str,inputs=[],radix="",toggle={},maxcap=0,fanout=0,load=0,peak=0,t1=0,t2=0,powertimedict={}):
        self.name=name
        self.cell=cell
        self.output=output
        self.inputs=inputs
        self.radix=radix
        self.toggle=toggle
        self.fanout=fanout
        self.load=load
        self.maxcap=maxcap
        self.powertimedict=powertimedict
        self.t1=t1
        self.t2=t2
        self.peak=peak