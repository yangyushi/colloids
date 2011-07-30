/*
 * multiscalefinder.hpp
 *
 *  Created on: 11 juil. 2011
 *      Author: mathieu
 */

#include "octavefinder.hpp"
#include <boost/ptr_container/ptr_vector.hpp>

#ifndef MULTISCALEFINDER_HPP_
#define MULTISCALEFINDER_HPP_

namespace Colloids {

class MultiscaleFinder : boost::noncopyable
{
public:

	virtual ~MultiscaleFinder();

	//accessors
	inline const size_t get_n_octaves() const {return this->octaves.size();};
	inline const OctaveFinder & get_octave(const size_t l) const {return *this->octaves[l];};
	virtual const size_t get_width() const =0;
	const size_t get_height() const {return this->octaves[0]->get_height()/2; };
	inline const size_t get_n_layers() const {return this->octaves[0]->get_n_layers();};
	const double & get_radius_preblur() const {return this->octaves[0]->get_radius_preblur();}
	const double & get_prefactor() const {return this->octaves[0]->get_prefactor();}
	void set_radius_preblur(const double &k=1.6);
	//processing
	std::vector<cv::Vec4d> operator()(const cv::Mat &input);
	virtual const cv::Vec3i previous_octave_coords(const cv::Vec4d &v) const;
	virtual const cv::Mat_<double> downscale(const size_t &o) = 0;
	virtual void seam(cv::Vec4d &v, const size_t &o) const =0;


protected:
	std::vector<OctaveFinder*> octaves;
	cv::Mat_<double> small, upscaled;
	MultiscaleFinder(){};
};

class MultiscaleFinder2D : public MultiscaleFinder
{
public:
	MultiscaleFinder2D(const int nrows=256, const int ncols=256, const int nbLayers=3, const double &preblur_radius=1.6);
	virtual const size_t get_width() const {return this->octaves[0]->get_width()/2; };
	virtual const cv::Mat_<double> downscale(const size_t &o);
	virtual void seam(cv::Vec4d &v, const size_t &o) const;
};

class MultiscaleFinder1D : public MultiscaleFinder
{
public:
	MultiscaleFinder1D(const int ncols=256, const int nbLayers=3, const double &preblur_radius=1.6);
	virtual const size_t get_width() const {return 1; };
	virtual const cv::Mat_<double> downscale(const size_t &o);
	virtual void seam(cv::Vec4d &v, const size_t &o) const;
};

}

#endif /* MULTISCALEFINDER_HPP_ */
