#define BOOST_TEST_DYN_LINK

#include "../graphic/tracker.hpp"
#include <boost/test/unit_test.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/progress.hpp>
#include <boost/array.hpp>
#include <set>

using namespace Colloids;
using namespace boost::posix_time;

//draw a sphere plane by plane on a 3D image
template<class T>
void drawsphere(boost::multi_array<T, 3> & input, const double &z, const double &y, const double &x, const double &r, const T & value=255)
{
	const double rsq = r*r;
	for(int k=std::max(0.0, z-r); k<std::min((double)input.shape()[0], z+r+1); ++k)
	{
		const double dz = pow(k-z, 2);
		for(int j=std::max(0.0, y-r); j<std::min((double)input.shape()[1], y+r+1); j++)
		{
			const double dy = pow(j-y, 2);
			for(int i=std::max(0.0, x-r); i<std::min((double)input.shape()[2], x+r+1); i++)
			{
				const double distsq = pow(i-x, 2) + dy + dz;
				if(distsq < rsq)
					input[k][j][i] = value;
			}

		}
	}
}

template<typename T>
void volume_shrink(const boost::multi_array<T, 3> & large, boost::multi_array<T, 3> & small, size_t factor)
{
	for(int k=0; k<small.shape()[0]; ++k)
		for(int j=0; j<small.shape()[1]; ++j)
			for(int i=0; i<small.shape()[2]; ++i)
			{
				double v = 0;
				for(size_t mk=0; mk<factor; ++mk)
					for(size_t mj=0; mj<factor; ++mj)
					{
						const T* a = &large[factor*k+mk][factor*j+mj][factor*i];
						v += std::accumulate(a, a+factor, 0);
					}
				small[k][j][i] = (v/(factor*factor*factor) + 0.5);
			}
}

BOOST_AUTO_TEST_SUITE( tracker )

BOOST_AUTO_TEST_SUITE( tracker_constructors )

    BOOST_AUTO_TEST_CASE( tracker_constructor_cube )
    {
        boost::array<size_t,3> dims = {64,64,64};
        Tracker tracker(dims);
        BOOST_CHECK_EQUAL(tracker.getDimensions()[0], 64);
        BOOST_CHECK_EQUAL(tracker.getDimensions()[1], 64);
        BOOST_CHECK_EQUAL(tracker.getDimensions()[2], 64);
        BOOST_CHECK_EQUAL(tracker.quiet, false);
        tracker.quiet = true;
        boost::array<double,3> 
            radiiMin = {4,4,4},
            radiiMax = {16,16,16};
        tracker.makeBandPassMask(radiiMin, radiiMax);
        //tracker.displayMask();
        BOOST_CHECK_EQUAL(tracker.FFTmask[0][0][0], false);
    }

BOOST_AUTO_TEST_SUITE_END() //tracker_constructors

BOOST_AUTO_TEST_SUITE( filling_tracking )

    BOOST_AUTO_TEST_CASE( filling_from_memory )
    {
        boost::array<size_t,3> dims = {64,64,64};
        Tracker tracker(dims);
        tracker.quiet = true;
        boost::multi_array<unsigned char, 3> input(boost::extents[64][64][64]); 
        boost::multi_array<float, 3> output(boost::extents[64][64][64]);

        // Assign values to the elements
        int values = 0;
        for(boost::multi_array<unsigned char, 3>::index i = 0; i != input.shape()[0]; ++i) 
            for(boost::multi_array<unsigned char, 3>::index j = 0; j != input.shape()[1]; ++j)
                for(boost::multi_array<unsigned char, 3>::index k = 0; k != input.shape()[2]; ++k)
                    input[i][j][k] = values++;
	    tracker.fillImage(input.origin());
	    tracker.copyImage(output.origin());
	    BOOST_CHECK_EQUAL_COLLECTIONS(
	        input.origin(), input.origin()+64*64*64,
	        output.origin(), output.origin()+64*64*64);
	    //tracker.display();
	    //draw a sphere
		drawsphere(input, 32, 32, 32, 4.0, (unsigned char)1);
		tracker.fillImage(input.origin());
		tracker.unpad();
		//tracker.display();
		
	}

	BOOST_AUTO_TEST_CASE( single_sphere )
	{
		boost::array<size_t,3> dims = {64,64,64};
        Tracker tracker(dims);
        tracker.quiet = true;
        boost::array<double,3> 
            radiiMin = {4,4,4},
            radiiMax = {16,16,16};
        tracker.makeBandPassMask(radiiMin, radiiMax);
        
        boost::multi_array<float, 3> input(boost::extents[64][64][64]);
        std::fill(input.origin(), input.origin()+64*64*64, 0);
		//draw a sphere
		drawsphere(input, 32, 32, 32, 4.0, 1.0f);
		tracker.fillImage(input.origin());
		
		tracker.FFTapplyMask();
		tracker.findPixelCenters(0.1f);
		BOOST_REQUIRE_EQUAL(std::accumulate(
		    tracker.centersMap.origin(), 
		    tracker.centersMap.origin()+64*64*64, 
		    0), 1);

		tracker.fillImage(input.origin());
		Particles pos = tracker.trackXYZ(0.1f);
		BOOST_REQUIRE_EQUAL(pos.size(), 1);
		//The particle should be at the center of the circle
		BOOST_CHECK_EQUAL(pos[0][0], 32);
		BOOST_CHECK_EQUAL(pos[0][1], 32);
		BOOST_CHECK_EQUAL(pos[0][2], 32);
	}

    BOOST_AUTO_TEST_CASE( subpix_positions )
	{
		boost::array<size_t,3> dims = {32,32,32};
        Tracker tracker(dims);
        tracker.quiet = true;
        tracker.view = false;
        boost::array<double,3> 
            radiiMin = {4,4,4},
            radiiMax = {16,16,16};
        tracker.makeBandPassMask(radiiMin, radiiMax);
		//cv cannot draw circle sizes better than a pixel, so the input image is drawn in high resolution
		boost::multi_array<float, 3> 
		    input(boost::extents[8*32][8*32][8*32]), 
		    small_input(boost::extents[32][32][32]);
		
		Particles v;
		for(int x=-4; x<4; ++x) //loop on position
		{
			for(int i=0; i<24; ++i) //loop on radius (4->7)
			{
				std::fill(input.origin(), input.origin()+64*64*64, 0);
				drawsphere(input, 8*16, 8*16, 8*16+x, 8*4+i);
				volume_shrink(input, small_input, 8);
				tracker.fillImage(small_input.origin());
				Particles v_s = tracker.trackXYZ(0.1f);
				BOOST_CHECK_MESSAGE(v_s.size()==1, "x="<<x/8.<<" r="<< 4+0.125*i<<"\t"<<v_s.size()<<" centers");
				std::copy(v_s.begin(), v_s.end(), std::back_inserter(v));
			}
		}
		BOOST_REQUIRE_EQUAL(v.size(), 8*24);
        
        v.exportToFile("test_output/subpix_positions3D");
		/*std::ofstream out("test_output/subpix_positions3D");
		for(size_t i=0; i<24; ++i)
		{
		    const double r = 4.0+0.125*i;
		    out<<r<<"\t";
			for(int x=-4; x<4; ++x)
			{
				BOOST_CHECK_CLOSE(v[i+24*(x+4)][2]-16, 16+0.125*x, 2);
				out<<v[i+24*(x+4)][2]<<"\t";
			}
			out<<"\n";
		}*/
	}
	
BOOST_AUTO_TEST_SUITE_END() //filling_tracking

BOOST_AUTO_TEST_SUITE_END() //tracker
