import os
directory="/home/sethuraman/Documents/Kahveci/RawDataSets/LeukemiaData/GSE33315_sorted/"
fileList=os.listdir(directory)
nfp=open("/home/sethuraman/Documents/Kahveci/ProbeToGeneMapper","r")
li=nfp.readlines()
probeMap=dict()
for each in li:
    spl=each[:-1].split('\t')
    if (len(spl)>1) and (len(spl[1])>0):
        probeMap[spl[0]]=spl[1]
li=[]
nfp.close()
comProbes=set()
vector=dict()
counter=0
while counter<len(fileList):
    fp=open(directory+fileList[counter],"r")
    d=dict()
    rd=[]
    rd=fp.readlines()
    #print "opening "+fileList[counter]+" %d"%(len(rd))
    outPutFilename=rd.pop(0)
    fp.close()
    for each in rd:
        l=each.split('\t')
        if probeMap.has_key(l[0][1:-1]):
            k=probeMap[l[0][1:-1]]
            val=float(l[1])
            if d.has_key(k):
                if (d[k]<val):
                    d[k]=val
            else :
                d[k]=val
    if counter==0:
        print "initailizing"
        comProbes=set(d.keys())
    else:
        #print "Adding %d size set"%(len(d.keys()))
        comProbes=comProbes.intersection(set(d.keys()))
    vector[fileList[counter]]=d
    counter+=1
wfp=open("/home/sethuraman/Documents/Kahveci/preprocessing/"+"Vector.txt","w+")
perList=vector.keys()
colList=list(comProbes)
print len(comProbes)
print " perList size:%d ColList size:%d"%(len(perList),len(colList))
clsfp=open("/home/sethuraman/Documents/Kahveci/RawDataSets/LeukemiaData/class.txt","r")
clslines=clsfp.readlines()
clsfp.close()
perCls=dict()
wfp.write("%d %d\n"%(len(perList),len(colList)))
for each in clslines:
          spl=each.strip().split("\t")
          perCls[spl[0].strip()]=spl[1].strip()
for person in perList:
          line=perCls[person]
          x=vector[person]
          for col in colList:
              line+=" %f"%(x[col])
          wfp.write("%s\n"%(line))
wfp.close()
