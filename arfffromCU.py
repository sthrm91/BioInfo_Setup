fp=open("1.train","r")
rd=fp.readlines()
wp=open("1_all_CU.arff","w")
wp.write("@RELATION CU\n")
n=rd[1].strip().split(' ')
for i in range(1,len(n)):
     wp.write("@ATTRIBUTE %d REAL\n"%(i))
wp.write("@ATTRIBUTE class {1,2}\n")
wp.write("@DATA\n")
wp.close()
for each in rd[1:]:
     w=each.strip().split()
     line=""
     for j in w[1:]:
             line+=j
             line+=","
     line+=w[0]
     wp.write("%s\n"%(line))
wp.close()

