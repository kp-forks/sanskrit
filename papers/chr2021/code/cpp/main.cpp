
#include "stdafx.h"

#include "types.h"
#include "helpers.h"
#include "settings.h"
#include "models.h"

bool read_assignments(const std::string &path, 
	std::vector<inttype> &Doc, std::vector<inttype> &Fea, std::vector<inttype> &Cit, std::vector<inttype>& Tim, std::vector<inttype> &Big)
{
	std::ifstream file(path, std::ios::binary);
	if (!file.good()) {
		return false;
	}
	int n = 0;
	std::string line;
	//const size_t maxr = 0;
#ifdef _DEBUG
	const size_t maxr = 0; //10000;
#else
	const size_t maxr = 0;
#endif
	while (std::getline(file, line)) {
		while (!line.empty() && (line.back() == 10 || line.back() == 13)) {
			line = line.substr(0, line.size() - 1);
		}
		if (!line.empty()){
			std::stringstream s(line);
			if (n==0) {
				Doc = numline<inttype>(s,maxr,-1);
			}else if (n == 1) {
				Fea = numline<inttype>(s,maxr,-1);
			}else if (n == 2) {
				Cit = numline<inttype>(s,maxr,-1);
			}else if(n==3){
				Tim = numline<inttype>(s,maxr,-1);
			}else if(n==4){
				Big = numline<inttype>(s,maxr,-1);
			}else{
				std::cout << "More than the expected five lines in the assignments. Exit." << std::endl;
				return false;
			}
		}
		n++;
	}
	return true;
}

/* just one training set */
int mainDefault()
{
	const std::string pathAffix = "noStop-noFrequent-withCitations-dateRanges-0-150.dat";
	const std::string inDir = "../data/input/";
	matrixf citmask = readMatrix<float>("../data/input/citation.mask");
	matrixf tau  = readMatrix<float>(inDir + "tau-ToCN-" + pathAffix);
	std::vector<inttype> Doc, Fea, Cit, Tim, Big;
	if(!read_assignments(inDir + "assignments-ToCN-" + pathAffix,Doc,Fea,Cit,Tim,Big)){
		std::cout << "assignments not loaded" << std::endl;
		return 0;
	}
	std::cout << "Read " << Tim.size() << " data points" << std::endl;
	std::cout << "T: " << *std::max_element(Tim.begin(),Tim.end()) << ", V: " << *std::max_element(Fea.begin(),Fea.end()) << std::endl;
	//return 0;
	const std::string respath = "../data/output/result-ToCN-" + pathAffix;
	const size_t nitrs = 100;
	std::map<std::string,float> params;
	cgsMmCitationNgram(Doc, Fea, Cit, Tim, Big,
		tau, citmask,
		respath, nitrs, params);
	return 0;
}



int main()
{
	return mainDefault();
}

