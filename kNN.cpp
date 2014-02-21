#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cstdlib>
using namespace std;

void shuffleArray(int *array,int size) 
{
	int n = size;
	while (n > 1) 
	{
		// 0 <= k < n.
		int k = rand()%n;		
		
		// n is now the last pertinent index;
		n--;					
		
		// swap array[n] with array[k]
		int temp = array[n];	
		array[n] = array[k];
		array[k] = temp;
	}
}


void returnVector(float *f, int*c, int p, int n, vector<int> noOfFeat, bool with_7, bool selectionMethod , vector< vector<int> >& result)
{
  //writes the incoming data to a temporary arff file and it is later supplied to weka for feature selection
  ofstream myfile_o;
  remove("example.arff");
  myfile_o.open ("example.arff");
  myfile_o << "@relation Temp_fold\n";
  int l=0;
  for(int i=0;i<p;i++)
  {
  myfile_o << "@attribute "<<i<<" numeric\n";
  }
  if(with_7)
  myfile_o << "@attribute class {1,2,3,4,5} \n";
  else
  myfile_o << "@attribute class {1,2,3,4,5,6} \n";
  myfile_o << "@data\n";
  l=0;
  // Actual data is written after this.
  for(int j=0;j<n*p;j++)
  {
  if(j%p==p-1)
  myfile_o<<f[j]<<", "<<c[l++]<<"\n";
  else 
  myfile_o<<f[j]<<", ";
  }
  myfile_o.close(); 
  //vector<vector<int> > result;
  vector<int> rank;
  for (int count =  noOfFeat.size()-1; count < noOfFeat.size(); count++)
  {    
   string line;
   char buffer [500];
   int n;
   //cout<<"generating for "<<noOfFeat[count]<<endl;
   if(selectionMethod)
   n=sprintf (buffer, "java -cp weka.jar weka.attributeSelection.SymmetricalUncertAttributeEval  -i example.arff  -c last  -s \"weka.attributeSelection.Ranker -N %d \" > feature.txt", noOfFeat[count]);
   else
    n=sprintf (buffer, "java -cp weka.jar weka.attributeSelection.InfoGainAttributeEval  -i example.arff  -c last  -s \"weka.attributeSelection.Ranker -N %d \" > feature.txt", noOfFeat[count]);
   system(buffer);
   ifstream myfile;
   myfile.open ("feature.txt");
   if (myfile.is_open())
   {
       while ( getline (myfile,line))
      {
         if(line.find("Selected attributes:")!=string::npos)
        {
            splitstring s(line);
            vector<string> flds = s.split(':');
            splitstring s1(flds[1]);
            flds.clear();       
	    flds=s1.split(',');
            for (int k = 0; k < flds.size(); k++)
                  rank.push_back(atoi(flds[k].c_str()));      
            //for (int k = 0; k < flds.size(); k++)
                //  cout << k << " => " << rank[k] << endl;
        }
     }
   for(int count = 0; count < noOfFeat.size(); count++)
    {
       vector<int> temp;
       for (int j=0;j<noOfFeat[count];j++)
        {  
           // cout << j << " => " << rank[j] << endl;     
            temp.push_back(rank[j]);  
         }  
       result.push_back(temp);
    }
myfile.close();
}
  else cout << "Unable to open file"; 
  remove("feature.txt");  
 }
return ;
}


void cross_validate(float *p,int p, int n)
{

int *index;
vector<int> test;
vector<int> train;
index=new int[n];

for(int i=0;i<n;i++)
index[i]=i;

shuffleArray(index,n);

  for(int i=0;i<10;i++)
  {

    for(int k=0;k<n;k++)
    {
    if((k>=10*i)&&(k<10*i+10))
      test.push_back(k);
    else
      train.push_back(k);
    }

    writeData(p,)

  }

}
