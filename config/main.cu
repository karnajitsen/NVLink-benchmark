#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#define MAXSTREAMNO 11
#include "config.h"

int main(int argc, char* argv[], char* envp[])
{

	// read config file with environment variable expansion support
	Config config("config/parameters.config", envp);
	std::ofstream fp;
	fp.open("parameters.h", std::ofstream::out);
	// basic usage of properties
        string chaseincsign,exthread,memadv,startomp, endomp,inromp, ompthreads, chelem, stride,repeat,dtype,dlattype,size,pattern,output,mmode,lat, tp, cudamem, kermemmap,stream,streamlist,startThread, endThread,itr;
 	// properties with expanded names (no difference in using)
	dtype =  config.pString("TPDATATYPE");
        chelem = config.pString("CHASE_ELEMENTS");
        dlattype =  config.pString("LATDATATYPE");
	size = config.pString("SIZE");
	pattern = config.pString("ACCESSPATTERN");
	kermemmap = config.pString("KER2MEMLOC");
	mmode = config.pString("MEMORYMODE");
        lat = config.pString("LATENCY");
        tp = config.pString("THROUGHPUT");
    	cudamem = config.pString("CUDAMEMCOPY");
	stream = config.pString("STREAM");
	repeat = config.pString("REPETATIONS");
	stride = config.pString("STRIDE");
	startThread = config.pString("START_THREAD");
	endThread = config.pString("END_THREAD");
	itr = config.pString("ITERATIONS");
        ompthreads = config.pString("OMP_THREADS");
        memadv = config.pString("MEM_ADVISE");
        exthread = config.pString("THREAD_EXPERIMENT");
	if(dtype.empty())
	dtype = "double";
	std::size_t pos = size.find(",");
	std::string range = size.substr(0,pos);
	std::string ifactorsize = size.substr(pos+1,size.length()-1);
	pos = range.find("-");
	std::string start = range.substr(0,pos);
	std::string end = range.substr(pos+1, range.length()-1);
	pos = start.find_first_of("BKMGP");
	std::string startsize = start.substr(0,pos);
	std::string startunit = start.substr(pos,start.length()-1);
	pos = end.find_first_of("BKMGP");
	std::string endsize = end.substr(0,pos);
	std::string endunit = end.substr(pos,end.length()-1);
	//cout << startsize << " " << startunit;
 //	pos = kermemmap.find(";");


        pos = stride.find(",");
        range = stride.substr(0,pos);
        std::string ifactorstride = stride.substr(pos+1,size.length()-1);
        pos = range.find("-");
        start = range.substr(0,pos);
        end = range.substr(pos+1, range.length()-1);
        pos = start.find_first_of("BKMGP");
        std::string startstride = start.substr(0,pos);
        std::string startstrideunit = start.substr(pos,start.length()-1);
        pos = end.find_first_of("BKMGP");
        std::string endstride = end.substr(0,pos);
        std::string endstrideunit = end.substr(pos,end.length()-1);

        pos = chelem.find(",");
        range = chelem.substr(0,pos);
        std::string left = chelem.substr(pos+1,size.length()-1);
       // std::string ifactorchelem = chelem.substr(pos+1,size.length()-1);
        pos = range.find("-");
        std::string startchelem = range.substr(0,pos);
        std::string endchelem = range.substr(pos+1, range.length()-1);
        pos = left.find(",");
        std::string ifactorchelem = left.substr(0,pos);
        chaseincsign = left.substr(pos+1,left.length()-1);
        


        pos = ompthreads.find(",");
        range = ompthreads.substr(0,pos);
        inromp = ompthreads.substr(pos+1,size.length()-1);
        pos = range.find("-");
        startomp = range.substr(0,pos);
        endomp= range.substr(pos+1, range.length()-1);

  	std::string kernelloc, memoryloc;
	kernelloc.append("{");
	memoryloc.append("{");
	int i,j;
	i = 0;
	j= 0;
	kermemmap.append(";");
	while(i < 6)
	{
		pos = kermemmap.find(";");	
		std::string temp = kermemmap.substr(0,pos);
		kermemmap = kermemmap.substr(pos+1, kermemmap.length());
		pos = temp.find("-");
		std::string kerloc = temp.substr(0,pos);
		kernelloc.append("\"");
		//if(kerloc != string::npos)
		kernelloc.append(kerloc);
		kernelloc.append("\"");
		std::string mmemloc = temp.substr(pos+1, temp.length());
		//pos=mmemloc.find(",");
		mmemloc.append(",");
		j = 0;
		memoryloc.append("{");
		while(j < 6) 
			{
				 pos = mmemloc.find(",");
 				  temp = mmemloc.substr(0,pos);
			  	  memoryloc.append("\"");
				  //if (temp != string::npos)
				  memoryloc.append(temp);
			          memoryloc.append("\"");
				 if (j < 5)
		       			memoryloc.append(",");
				 j++;
				 mmemloc = mmemloc.substr(pos+1, mmemloc.length());

			}
		memoryloc.append("}");
		if(i<5)
		{
			kernelloc.append(",");
			memoryloc.append(",");
		}
		i++;
	}
	memoryloc.append("};");
	kernelloc.append("};");
        std::string temp;
        int noofstream = 0;
     streamlist.append("{");
     while(noofstream < MAXSTREAMNO)
	 {
    		pos = stream.find(",");
                if (pos == string::npos)
                {
 		streamlist.append("\"");
                streamlist.append(stream.substr(0,stream.length()));
                streamlist.append("\"");
                noofstream++;
                break;
                }
                else{
                streamlist.append("\"");
                streamlist.append(stream.substr(0,pos));
                streamlist.append("\",");
		stream = stream.substr(pos+1, stream.length());
                }
        noofstream++;
        } 
    streamlist.append("}"); 

 

    
	fp << "#include<iostream>" << std::endl;
	fp << "#include <string>" << std::endl;
	fp << "#include <vector>" << std::endl;        
        fp << "#define NOOFSTREAM " << noofstream << std::endl;
        fp << "#define MAXKERMEMMAPS 6" << std::endl;
	fp << "using namespace std;" << std::endl;
       	fp << std::endl;	
	fp << "typedef " << dtype << " dType;" << std::endl;
        fp << "typedef " << dlattype << " dLatType;" << std::endl;
	fp << "const std::string accPattern = \"" << pattern << "\";" << std::endl;
	fp << "const std::string memMode = \"" << mmode << "\";" << std::endl;
    	 fp << "const char latMode =  '" << lat << "';" << std::endl;
	fp << "const char memAdvise =  '" << memadv << "';" << std::endl;
	 fp << "const char tpMode = '" << tp << "';" << std::endl;
	 fp << "const char cudaMemMode = '" << cudamem << "';" << std::endl;
	fp << "const char exp_thread = '" << exthread << "';" << std::endl;
	fp << "const int startStride = " << startstride << ";" << std::endl;
        fp << "const char startStrideUnit = \'" << startstrideunit << "\';" << std::endl;
        fp << "const int endStride = " << endstride << ";" << std::endl;
        fp << "const char endStrideUnit = \'" << endstrideunit << "\';" << std::endl;
	
 	 fp << "const int startChelem = " << startchelem << ";" << std::endl;
  	 fp << "const int endChelem = " << endchelem << ";" << std::endl;
        fp << "const char chaseIncSign = \'" << chaseincsign << "\';" << std::endl;
	
	fp << "const int startSize = " << startsize << ";" << std::endl;
	fp << "const char startUnit = \'" << startunit << "\';" << std::endl;
        fp << "const int endSize = " << endsize << ";" << std::endl;
	fp << "const char endUnit = \'" << endunit << "\';" << std::endl;
         fp << "int startomp = " << startomp << ";" << std::endl;
         fp << "int endomp = " << endomp << ";" << std::endl;
	

        fp << "const int inromp = " << inromp << ";" << std::endl;

	fp << "const int inc_size = " << ifactorsize << ";" << std::endl;
        fp << "const int inc_stride = " << ifactorstride << ";" << std::endl;
        fp << "const int inc_chelem = " << ifactorchelem << ";" << std::endl;
	
	fp << "const std::string kloc[6] = " << kernelloc << std::endl;
	fp << "unsigned int num_times = " << repeat << ";" << std::endl;
	fp << "const std::string mloc[6][6] = " << memoryloc << std::endl;
	fp << "std::string  strMode[" << noofstream << "] = " << streamlist << ";" << std::endl;
	fp << "const int startBlock = " << startThread << ";" << std::endl;
	fp << "const int endBlock = " << endThread << ";" << std::endl;
	fp << "const int iterations = " << itr << ";" << std::endl;
        fp << "std::vector<string> streamMode;" << std::endl;
	fp.close();
	return 0;
}

