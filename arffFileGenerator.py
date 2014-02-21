import os
#Specify the directory of the input files here
directory="/home/sethuraman/Documents/SVM/KahveciLab/LeukemiaData/GSE33315_sorted/"
fileList=os.listdir(directory)
#Specify the file for which has the mapping from probes to genes
nfp=open("/home/sethuraman/Documents/SVM/KahveciLab/LeukemiaData/ProbeToGeneMapper","r")
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
        # checks if the probe exists in the probe map otherwise ignored
        if probeMap.has_key(l[0][1:-1]):
            k=probeMap[l[0][1:-1]]
            val=float(l[1])
            # checks if the probe has a greater value than the existing one. If yes then updated otherwise ignored
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
# writing to the output file
wfp=open("/home/sethuraman/Documents/SVM/KahveciLab/preprocessing/"+"Vector_without_7.arff","w+")
perList=vector.keys()
colList=list(comProbes)
print len(comProbes)
clsfp=open("/home/sethuraman/Documents/SVM/KahveciLab/LeukemiaData/class.txt","r")
perCls=dict()
clslines=clsfp.readlines()
for each in clslines:
          spl=each.strip().split("\t")
          if spl[1].strip() in ["1","2","3","4","8"]:
              perCls[spl[0].strip()]=spl[1].strip()
clsfp.close()
print " perList size:%d ColList size:%d"%(len(perCls),len(colList))
#wfp.write("%d %d\n"%(len(colList),len(perCls)))
wfp.write("@relation LeukemiaData\n")
for one in colList:
    wfp.write("@attribute %s real\n"%(one))
wfp.write("@attribute class {1,2,3,4,8}\n")
wfp.write("@data\n")
for person in perCls.keys():
          line=""
          x=vector[person]
          for col in colList:
                line+="%f, "%(x[col])
          line+=perCls[person]
          wfp.write("%s\n"%(line))
wfp.close()
