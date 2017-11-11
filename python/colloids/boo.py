#
#    Copyright 2011 Mathieu Leocmach
#
#    This file is part of Colloids.
#
#    Colloids is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Colloids is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Colloids.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np
from scipy.special import sph_harm
try:
    from scipy import weave
    from scipy.weave import converters
except ImportError:
    try:
        import weave
        from weave import converters
    except ImportError:
        pass
import numexpr
import numba
from colloids import periodic

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2
        

def cart2sph(cartesian):
    """Convert Cartesian coordinates [[x,y,z],] to spherical coordinates [[r,phi,theta],]
phi is cologitudinal and theta azimutal"""
    spherical = np.zeros_like(cartesian)
    #distances
    c2 = cartesian**2
    r2 = c2.sum(-1)
    spherical[:,0] = np.sqrt(r2)
    #work only on non-zero, non purely z vectors
    sel = (r2 > c2[:,0]) | (r2+1.0 > 1.0)
    x, y, z = cartesian[sel].T
    r = spherical[sel,0]
    #colatitudinal phi [0, pi[
    spherical[sel,1] = np.arccos(z/r)
    #azimutal (longitudinal) theta [0, 2pi[
    theta = np.arctan2(y, x)
    theta[theta<0] += 2*np.pi
    spherical[sel,2] = theta
    return spherical
    
def vect2Ylm(v, l):
    """Projects vectors v on the base of spherical harmonics of degree l."""
    spherical = cart2sph(v)
    return sph_harm(
        np.arange(l+1)[:,None], l, 
        spherical[:,2][None,:], 
        spherical[:,1][None,:]
        )
        
def single_pos2qlm(pos, i, ngb_indices, l=6):
    """Returns the qlm for a single position"""
    #vectors to neighbours
    vectors = pos[ngb_indices]-pos[i]
    return vect2Ylm(vectors, l).mean(-1)
    
def bonds2qlm(pos, bonds, l=6, periods=-1):
    """Returns the qlm for every particle"""
    qlm = np.zeros((len(pos), l+1), np.complex128)
    #spherical harmonic coefficients for each bond
    Ylm = vect2Ylm(
        periodic.periodify(
            pos[bonds[:,0]],
            pos[bonds[:,1]],
            periods
        ),
        l
    ).T
    #bin bond into each particle belonging to it
    np.add.at(qlm, bonds[:,0], Ylm)
    np.add.at(qlm, bonds[:,1], Ylm)
    #divide by the number of bonds each particle belongs to
    Nngb = np.zeros(len(pos), int)
    np.add.at(Nngb, bonds.ravel(), 1)
    return qlm / np.maximum(1, Nngb)[:,None]
    
def coarsegrain_qlm(qlm, bonds, inside):
    """Coarse grain the bond orientational order on the neighbourhood of a particle
    $$Q_{\ell m}(i) = \frac{1}{N_i+1}\left( q_{\ell m}(i) +  \sum_{j=0}^{N_i} q_{\ell m}(j)\right)$$
    Returns Qlm and the mask of the valid particles
    """
    #Valid particles must be valid themselves have only valid neighbours
    inside2 = np.copy(inside)
    np.bitwise_and.at(inside2, bonds[:,0], inside[bonds[:,1]])
    np.bitwise_and.at(inside2, bonds[:,1], inside[bonds[:,0]])
    #number of neighbours
    Nngb = np.zeros(len(qlm), int)
    np.add.at(Nngb, bonds.ravel(), 1)
    #sum the boo coefficients of all the neighbours
    Qlm = np.zeros_like(qlm)
    np.add.at(Qlm, bonds[:,0], qlm[bonds[:,1]])
    np.add.at(Qlm, bonds[:,1], qlm[bonds[:,0]])
    Qlm[np.bitwise_not(inside2)] = 0
    return Qlm / np.maximum(1, Nngb)[:,None], inside2
    
    
def boo_product(qlm1, qlm2):
    """Product between two qlm"""
    n = np.atleast_2d(numexpr.evaluate(
        """real(complex(real(a), -imag(a)) * b)""",
        {'a':qlm1, 'b':qlm2}
        ))
    p = numexpr.evaluate(
        """4*pi/(2*l+1)*(2*na + nb)""",
        {
            'na': n[:,1:].sum(-1),
            'nb': n[:,0],
            'l': n.shape[1]-1,
            'pi': np.pi
            })
    return p

def ql(qlm):
    """Second order rotational invariant of the bond orientational order of l-fold symmetry
    $$ q_\ell = \sqrt{\frac{4\pi}{2l+1} \sum_{m=-\ell}^{\ell} |q_{\ell m}|^2 } $$"""
    q = abs2(qlm[:,0])
    q += 2*abs2(qlm[:,1:]).sum(-1)
    l = qlm.shape[1]-1
    return np.sqrt(4*np.pi / (2*l+1) * q)
    
def wl(qlm):
    """Third order rotational invariant of the bond orientational order of l-fold symmetry
    $$ w_\ell = \sum_{m_1+m_2+m_3=0} 
			\left( \begin{array}{ccc}
				\ell & \ell & \ell \\
				m_1 & m_2 & m_3 
			\end{array} \right)
			q_{\ell m_1} q_{\ell m_2} q_{\ell m_3}
			$$"""
    l = qlm.shape[1]-1
    w = np.zeros(qlm.shape[0], qlm.dtype)
    for m1 in range(-l, l+1):
        for m2 in range(-l, l+1):
            m3 = -m1-m2
            if -l<=m3 and m3<=l:
                w+= get_w3j(l, [m1, m2, m3]) * get_qlm(qlm, m1) * get_qlm(qlm, m2) * get_qlm(qlm, m3)
    return w.real
    
def get_qlm(qlms, m):
    if m>=0:
        return qlms[:,m]
    if (-m)%2 == 0:
        return np.conj(qlms[:,-m])
    return -np.conj(qlms[:,-m])

def gG_l(pos, qlms, Qlms, is_center, Nbins, maxdist):
    """Spatial correlation of the qlms and the Qlms (non normalized.
    For each particle tagged as is_center, do the cross product between their qlm, their Qlm and count, 
    then bin each quantity with respect to distance. 
    The two first sums need to be normalised by the last one.
    
     - pos is a Nxd array of coordinates, with d the dimension of space
     - qlm is a Nx(2l+1) array of boo coordinates for l-fold symmetry
     - Qlm is the coarse-grained version of qlm
     - is_center is a N array of booleans. For example all particles further away than maxdist from any edge of the box.
     - Nbins is the number of bins along r
     - maxdist is the maximum distance considered"""
    assert len(pos) == len(qlms)
    assert len(qlms) == len(Qlms)
    assert len(is_center) == len(pos)
    #conversion factor between indices and bins
    l2r = Nbins/maxdist
    #result containers
    hqQ = np.zeros((Nbins,2)) 
    g = np.zeros(Nbins, int)
    #spatial indexing
    tree = KDTree(pos, 12)
    centertree = KDTree(pos[is_center], 12)
    #all pairs of points closer than maxdist with their distances in a record array
    query = centertree.sparse_distance_matrix(tree, maxdist, output_type='ndarray')
    #keep only pairs where the points are distinct
    centerindex = np.where(is_center)[0]
    query['i'] = centerindex[query['i']]
    good = query['i'] != query['j']
    query = query[good]
    #binning of distances
    rs = (query['v'] * l2r).astype(int)
    np.add.at(g, rs, 1)
    #binning of boo cross products
    pqQs = np.empty((len(rs),2))
    pqQs[:,0] = boo_product(qlms[query['i']], qlms[query['j']])
    pqQs[:,1] = boo_product(Qlms[query['i']], Qlms[query['j']])
    np.add.at(hqQ, rs, pqQs)
    return hqQ[:,0], hqQ[:,1], g

def periodic_gG_l(pos, L, qlms, Qlms, Nbins):
    """
    Spatial correlation of the qlms and the Qlms in a periodic box of size L
    """
    assert len(pos) == len(qlms)
    assert len(qlms) == len(Qlms)
    maxdist = L/2.0
    maxsq = float(maxdist**2)
    hQ = np.zeros(Nbins)
    hq = np.zeros(Nbins)
    g = np.zeros(Nbins, int)
    code = """
    #pragma omp parallel for
    for(int i=0; i<Npos[0]-1; ++i)
    {
        for(int j=i+1; j<Npos[0]; ++j)
        {
            if(i==j) continue;
            double disq = 0.0;
            for(int dim=0; dim<3;++dim)
                disq += pow(periodic_dist(pos(i,dim), pos(j,dim), L), 2);
            if(disq>=(double)maxsq)
                continue;
            const int r = sqrt(disq/(double)maxsq)*Nbins;
            double pq = real(qlms(i,0)*conj(qlms(j,0)));
            for(int m=1; m<Nqlms[1]; ++m)
                pq += 2.0*real(qlms(i,m)*conj(qlms(j,m)));
            pq *= 4.0*M_PI/(2.0*(Nqlms[1]-1)+1);
            double pQ = real(Qlms(i,0)*conj(Qlms(j,0)));
            for(int m=1; m<NQlms[1]; ++m)
                pQ += 2.0*real(Qlms(i,m)*conj(Qlms(j,m)));
            pQ *= 4.0*M_PI/(2.0*(NQlms[1]-1)+1);
            #pragma omp critical
            {
                ++g(r);
                hq(r) += pq;
                hQ(r) += pQ;
            }
        }
    }
    """
    weave.inline(
        code,['qlms', 'Qlms', 'pos', 'maxsq', 'Nbins', 'hQ', 'hq', 'g', 'L'],
        type_converters =converters.blitz,
        support_code = periodic.dist_code,
        extra_compile_args =['-O3 -fopenmp'],
        extra_link_args=['-lgomp'],
        verbose=2, compiler='gcc')
    return hq, hQ, g
    
def steinhardt_g_l(pos, bonds, is_center, Nbins, maxdist, l=6):
    """
    Spatial correlation of the bond's spherical harmonics
    """
    assert len(is_center) == len(pos)
    maxsq = float(maxdist**2)
    hq = np.zeros(Nbins)
    g = np.zeros(Nbins, int)
    qlms = np.zeros([len(bonds), l+1], np.complex128)
    bpos = np.zeros([len(bonds), 3])
    code = """
    //position and spherical harmonics of each bond
    #pragma omp parallel for
    for(int b=0; b<Nbonds[0]; ++b)
    {
        int i = bonds(b,0), j = bonds(b,1);
        blitz::Array<double,1> cart(pos(i, blitz::Range::all()) - pos(j, blitz::Range::all()));
        bpos(b, blitz::Range::all()) = pos(j, blitz::Range::all()) + 0.5 * cart;
        double sph[3] = {0, 0, 0};
        sph[0] = sqrt(blitz::sum(blitz::pow(cart, 2)));
        if(abs(cart(2))==sph[0] || sph[0]*sph[0]+1.0 == 1.0)
        {
            sph[1] = 0;
            sph[2] = 0;
        }
        else
        {
            sph[1] = acos(cart(2)/sph[0]);
            sph[2] = atan2(cart(1), cart(0));
            if(sph[2]<0)
                sph[2] += 2.0*M_PI;
        }
        for(int m=0; m<Nqlms[1]; ++m)
            qlms(b,m) = boost::math::spherical_harmonic(Nqlms[1]-1, m, sph[1], sph[2]);
    }
    #pragma omp parallel for
    for(int b=0; b<Nbonds[0]; ++b)
    {
        int i = bonds(b,0), j = bonds(b,1);
        if(!is_center(i) || !is_center(j))
            continue;
        for(int c=0; c<Nbonds[0]; ++c)
        {
            const double disq = blitz::sum(blitz::pow(bpos(b, blitz::Range::all()) - bpos(c, blitz::Range::all()),2));
            if(disq>=(double)maxsq)
                continue;
            const int r = sqrt(disq/(double)maxsq)*Nbins;
            double pq = real(qlms(b,0)*conj(qlms(c,0)));
            for(int m=1; m<Nqlms[1]; ++m)
                pq += 2.0*real(qlms(b,m)*conj(qlms(c,m)));
            pq *= 4.0*M_PI/(2.0*(Nqlms[1]-1)+1);
            #pragma omp critical
            {
                ++g(r);
                hq(r) += pq;
            }
        }
    }
    """
    weave.inline(
        code,['qlms', 'pos', 'bonds', 'bpos', 'maxsq', 'Nbins', 'hq', 'g','is_center'],
        type_converters =converters.blitz,
        headers=['<boost/math/special_functions/spherical_harmonic.hpp>'],
        extra_compile_args =['-O3 -fopenmp'],
        extra_link_args=['-lgomp'],
        verbose=2, compiler='gcc')
    return hq, g
            
_w3j = [
    [1],
    np.sqrt([2/35., 1/70., 2/35., 3/35.])*[-1,1,1,-1],
    np.sqrt([
        2/1001., 1/2002., 11/182., 5/1001.,
        7/286., 5/143., 14/143., 35/143., 5/143.,
        ])*[3, -3, -1/3.0, 2, 1, -1/3.0, 1/3.0, -1/3.0, 1],
    np.sqrt([
        1/46189., 1/46189.,
        11/4199., 105/46189.,
        1/46189., 21/92378.,
        1/46189., 35/46189., 14/46189.,
        11/4199., 21/4199., 7/4199.,
        11/4199., 77/8398., 70/4199., 21/4199.
        ])*[-20, 10, 1, -2, -43/2.0, 3, 4, 2.5, -6, 2.5, -1.5, 1, 1, -1, 1, -2],
    np.sqrt([
        10/96577., 5/193154.,
        1/965770., 14/96577.,
        1/965770., 66/482885.,
        5/193154., 3/96577., 77/482885.,
        65/14858., 5/7429., 42/37145.,
        65/14858., 0.0, 3/7429., 66/37145.,
        13/74290., 78/7429., 26/37145., 33/37145.,
        26/37145., 13/37145., 273/37145., 429/37145., 11/7429.,
        ])*[
            7, -7, -37, 6, 73, -3,
            -5, -8, 6, -1, 3, -1,
            1, 0, -3, 2, 7, -1, 3, -1,
            1, -3, 1, -1, 3],
    np.sqrt([
        7/33393355., 7/33393355.,
        7/33393355., 462/6678671.,
        7/33393355., 1001/6678671.,
        1/233753485., 77/6678671., 6006/6678671.,
        1/233753485., 55/46750697., 1155/13357342.,
        1/233753485., 2926/1757545., 33/46750697., 3003/6678671.,
        119/1964315., 22/2750041., 1914/474145., 429/5500082.,
        17/13750205., 561/2750041., 77/392863., 143/27500410., 2002/392863.,
        323/723695., 1309/20677., 374/144739., 143/144739., 1001/206770.,
        323/723695., 7106/723695., 561/723695., 2431/723695., 2002/103385., 1001/103385.
        ])*[
            -126, 63, 196/3.0, -7, -259/2.0, 7/3.0,
            1097/3.0, 59/6.0, -2,
            4021/6.0, -113/2.0, 3,
            -914, 1/3.0, 48, -3,
            -7/3.0, 65/3.0, -1, 3,
            214/3.0, -3, -2/3.0, 71/3.0, -1,
            3, -1/3.0, 5/3.0, -2, 1/3.0,
            2/3.0, -1/3.0, 2, -4/3.0, 2/3.0, -1]
    ]
_w3j_m1_offset = np.array([0,1,2,4,6,9,12,16,20,25,30], int)

def get_w3j(l, ms):
    sm = np.sort(np.abs(ms))
    return _w3j[l/2][_w3j_m1_offset[sm[-1]]+sm[0]]
