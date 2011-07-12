/*
 * multiscalefinder.cpp
 *
 *  Created on: 11 juil. 2011
 *      Author: mathieu
 */

#include "multiscalefinder.hpp"
#include <stdexcept>
#include <iostream>

using namespace std;

namespace Colloids {

	MultiscaleFinder::MultiscaleFinder(const int nrows, const int ncols, const int nbLayers, const double &preblur_radius) :
			small(nrows, ncols),
			upscaled(2*nrows, 2*ncols)
	{
		this->octaves.reserve((size_t)min(log(nrows/12.0)/log(2), log(ncols/12.0)/log(2)));
		this->octaves.push_back(new OctaveFinder(2*nrows, 2*ncols, nbLayers, preblur_radius));
		int ocr = nrows, occ = ncols;
		while(ocr >= 12 && occ >= 12)
		{
			this->octaves.push_back(new OctaveFinder(ocr, occ, nbLayers, preblur_radius));
			ocr /= 2;
			occ /= 2;
		}

	}

	MultiscaleFinder::~MultiscaleFinder() {
		// TODO Auto-generated destructor stub
	}

    void MultiscaleFinder::set_radius_preblur(const double & k)
    {
    	for(size_t o=0; o<this->octaves.size(); ++o)
    		this->octaves[o]->set_radius_preblur(k);
    }

    std::vector<cv::Vec4d> MultiscaleFinder::operator ()(const cv::Mat & input)
    {
    	if(input.rows != (int)this->get_width())
    		throw std::invalid_argument("MultiscaleFinder::operator () : the input's rows must match the width of the finder");
    	if(input.cols != (int)this->get_height())
    	    throw std::invalid_argument("MultiscaleFinder::operator () : the input's cols must match the height of the finder");

    	input.convertTo(small, this->small.type());
    	//upscale the input to fill the first octave
    	//cv::resize does not work with double input, so we do it by hand
    	for(int j=0; 2*j<this->upscaled.cols; ++j)
    		for(int i=0; 2*i<this->upscaled.rows; ++i)
    			this->upscaled(2*i, 2*j) = small(i,j);
    	for(int j=0; 2*j<this->upscaled.cols; ++j)
			for(int i=0; 2*i+1<this->upscaled.rows; ++i)
				this->upscaled(2*i+1, 2*j) = small(i,j);
    	for(int j=0; 2*j+1<this->upscaled.cols; ++j)
			for(int i=0; 2*i<this->upscaled.rows; ++i)
				this->upscaled(2*i, 2*j+1) = small(i,j);
    	for(int j=0; 2*j+1<this->upscaled.cols; ++j)
			for(int i=0; 2*i+1<this->upscaled.rows; ++i)
				this->upscaled(2*i+1, 2*j+1) = small(i,j);

    	std::cout<<"upscale ok"<<std::endl;
    	std::vector<cv::Vec4d> centers = (*this->octaves[0])(this->upscaled, true);
    	std::cout<<"octave 0 ok"<<std::endl;
    	for(size_t c=0; c<centers.size(); ++c)
    		for(size_t i=0; i<3;++i)
    			centers[c][i] /= 2.0;
    	std::cout<<"scaling octave 0 ok"<<std::endl;
    	//Octave 1 corresponds to the size of the input image.
    	//To avoid errors in the upsampling+downsampling process, we use the input directly
    	if(this->octaves.size()>1)
    	{
    		std::cout<<"Ready for octave 1"<<std::endl;
    		std::vector<cv::Vec4d> v = (*this->octaves[1])(input, true);
    		std::cout<<"octave 1 ok"<<std::endl;
    		centers.reserve(centers.size() + v.size());
    		std::copy(v.begin(), v.end(), std::back_inserter(centers));
    		std::cout<<"adding octave 1 ok"<<std::endl;
    	}

    	for(size_t o=2; o<this->octaves.size(); ++o)
    	{
    		//second to last Gaussian layer of octave o-1 has a blurring radius two time larger than the original
    		cv::Mat_<double> roi2 = small(
    				cv::Range(0, this->octaves[o]->get_width()),
    				cv::Range(0, this->octaves[o]->get_height())
    		);
    		const cv::Mat_<double> & a = this->octaves[o-1]->get_layersG(this->octaves[o-1]->get_n_layers());
    		for(int j=0; j<roi2.rows; ++j)
    			for(int i=0; i<roi2.cols; ++i)
    				roi2(i,j) = (a(2*i, 2*j) + a(2*i+1, 2*j) + a(2*i, 2*j+1) + a(2*i+1, 2*j+1))/4.0;
    		std::vector<cv::Vec4d> v = (*this->octaves[o])(roi2);
    		centers.reserve(centers.size() + v.size());
    		for(size_t c=0; c<v.size(); ++c)
    		{
    			for(size_t i=0; i<3;++i)
    				v[c][i] *= pow(2, o-1);
    			centers.push_back(v[c]);
    		}
    		std::cout<<"octave "<<o<<" ok"<<std::endl;
    	}

    	return centers;
    }

}