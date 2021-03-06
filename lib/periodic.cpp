/**
    Copyright 2008,2009 Mathieu Leocmach

    This file is part of Colloids.

    Colloids is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Colloids is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Colloids.  If not, see <http://www.gnu.org/licenses/>.
**/

#include "periodic.hpp"

using namespace std;
using namespace Colloids;

/** \brief Transform a vector for it to be in the boundary conditions */
void PeriodicParticles::periodify(Coord &v) const
{
    for(size_t i=0;i<3;++i)
    {
        if(v[i]>getPeriod(i)/2.0) v[i] -= getPeriod(i);
        if(v[i]<=-getPeriod(i)/2.0) v[i] += getPeriod(i);
    }
}


/** \brief get the difference vector between a position and one of the particles */
Coord PeriodicParticles::getDiff(const Coord &from,const size_t &to) const
{
    Coord diff(3);
    diff = Particles::getDiff(from,to);
    periodify(diff);
    return diff;
}
/** \brief get the difference vector between two particles */
Coord PeriodicParticles::getDiff(const size_t &from,const size_t &to) const
{
    Coord diff(3);
    diff = Particles::getDiff(from,to);
    periodify(diff);
    return diff;
}

/** \brief return the number density without margin */
double PeriodicParticles::getNumberDensity() const
{
    return size()/bb.area();
}

/**
    \brief get the index of the particle contained inside a reduction of the bounding box.
    \param cutoff range to exclude from each side of the box
    \return list of all the particles
     Dummy function in the case of periodic boundary condition.
*/
vector<size_t> PeriodicParticles::selectInside(const double &margin, const bool noZ) const
{
    vector<size_t> inside(size());
    for(size_t i=0;i<size();++i)
        inside[i]=i;
    return inside;
}

/**
    \brief get the index of the particles enclosed inside a given bounding box, with periodicity
    \param b search range
    \return list of the index
*/
vector<size_t> PeriodicParticles::selectEnclosed(const BoundingBox &b) const
{
    //case where the query doesn't get across the boundaries
    if(bb.encloses(b))
        return Particles::selectEnclosed(b);

    // case where the periodicity has to be taken into account
    BoundingBox queryBox = b;
    valarray<double> translation(0.0,3);
    list<size_t>total;
    vector<size_t>newParts;
    for(int i=-1;i<=1;++i)
    {
        translation[0]=i*getPeriod(0);
        for(int j=-1;j<=1;++j)
        {
            translation[1]=j*getPeriod(1);
            for(int k=-1;k<=1;++k)
            {
                translation[2]=k*getPeriod(2);
                //the query box get translated to an other side of the periodic boundaries
                for(size_t d=0;d<3;++d)
                {
                    queryBox.edges[d].first=b.edges[d].first+translation[d];
                    queryBox.edges[d].second=b.edges[d].second+translation[d];
                }
                //if ever the box contains no particle, the query on the R*Tree return an empty result in constant time (immediately)
                //so no special precaution to discard in advance irrealistic translations of the query box.
                newParts = Particles::selectEnclosed(queryBox);
                //the new set of particles is added
                copy(newParts.begin(),newParts.end(), back_inserter(total));
            }
        }
    }
    total.sort();
    total.unique();
    return vector<size_t>(total.begin(), total.end());

}
