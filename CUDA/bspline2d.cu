#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

#ifndef MAX_DEGREE 
#define MAX_DEGREE 9 // supports p, q up to this value 
#endif 
#ifndef MAX_DERIV 
#define MAX_DERIV 2 // calculate up to 2nd derivative (0, 1, 2)
#endif 

using Real = double; 


// Host functions 
static inline void generate_knots_and_colloc_pts_host( 
    int p, int num_basis, Real xmin, Real xmax,
    Real stretch_factor, 
    std::vector<Real>& knots_out, 
    std::vector<Real>& colloc_pts_out)
{
  const int m = num_basis + p + 1; // knots 
  knots_out.assign(m, Real(0)); // fill with zero 

  // clamped ends 
  for (int i = 0; i <= p; ++i) knots_out[i] = Real(0); 
  for (int i = num_basis; i < m; ++i) knots_out[i] = Real(1); 

  // interior 
  for (int i = p + 1; i < num_basis; ++i)
  {
    Real s = Real(i - p) / Real(num_basis - p);
    if (std::abs(stretch_factor) < 1e-12)
    {
      knots_out[i] = s; // uniform 
    } 
    else 
    {
      knots_out[i] = Real(0.5) * (std::tanh(stretch_factor * (2.0 * s - 1.0)) + 1.0);
    }
  }

  // Greville abscissae (normalized [0,1])
  std::vector<Real> colloc_norm(num_basis, Real(0));
  for (int i = 0; i < num_basis; ++i)
  {
    Real sum = 0; 
    for (int k = i + 1; k <= i + p; ++k) sum += knots_out[k];
    colloc_norm[i] = sum / Real(p);
  }

  // Map to physical domain
  colloc_pts_out.resize(num_basis); 
  for (int i = 0; i < num_basis; ++ i)
  {
    colloc_pts_out[i] = xmin + colloc_norm[i] * (xmax - xmin);
  }

  // snap endpoints 
  colloc_pts_out.front()  = xmin; 
  colloc_pts_out.back()   = xmax;
}

// Device functions 

// find the spanning index of a given collocation point
__device__ inline int find_span_norm( 
  Real xi, int p, const Real* __restrict__ knots, int n)
{
  // Handle xi ==1 case 
  if (xi >= Real(1) - Real(1e-14)) return n; 

  // binary search 
  int low = p;
  int high = n + 1; 
  int mid = (low + high) >> 1; // divide by 2 

  while (xi < knots[mid] || xi >= knots[mid + 1])
  {
    if (xi < knots[mid])  high  = mid; 
    else              low   = mid; 
    mid = (low + high) >> 1;
  }
  return mid; 
}

// For a known span, the non-zero basis functions and their derivatives up to order nd is computed. 
__device__ inline eval_basis_and_derivatives(
  int spanIndex, Real xi, int p, const Real* __restrict__ knots, int maxDerivOrder,
  Real derivOut[(MAX_DEGREE + 1)][(MAX_DEGREE + 1)])  // output: derivatives and values 
{
  Real ndu[MAX_DEGREE + 1][MAX_DEGREE + 1];         // holds the basis value as a triangular table 
  Real a[2][MAX_DEGREE + 1];                        // will be used for derivative back-susbtitution
  Real left[MAX_DEGREE + 1], right[MAX_DEGREE + 1];

  ndu[0][0] = 1.0;

  // build the triangle column-by column
  for (int j = 1; j <= p; ++j)
  {
    left[j]   = xi - knots[spanIndex + 1 - j];
    right[j]  = knots[spanIndex + j] - xi; 
    Real sum  = 0.0; 
    for (int r = 0; r < j; ++ r)
    {
      Real denom  = right[r + 1] + left[j - r];
      Real common = ndu[r][j - 1] / denom; 

      ndu[r][j] = sum + right[r + 1] * common;
      sum       = left[j - r] * common;
    }
    ndu[j][j] = sum
  }
  // ndu[r][j] hols the Cox-de Boor triangle of basis values, where column j holds degree j values for the j + 1 nonzero local bases.

  // basis values are last column
  for (int j = 0; j <= p; ++p)
    derivOut[0][j]  = ndu[j][p];

  // compute derivatives (orders k = 1..maxDerivOrder) for each local basis r (0..p)
  for (int r = 0; r<=p; ++r)
  {
    int s1 = 0, s2 = 1; // which row of 'a' we read from (s1) and write to (s2)
    a[0][0] = 1.0;
    // loop over derivative order k = 1..maxDerivOrder
    for (int k = 1; k <= maxDerivOrder; ++k)
    {
      Real d  = 0.0;    // holds the values for the k-th derivative of local basis r 
      int rk  = r - k;  // r shifted down by derivative order
      int pk  = p - k;  // degree p reduced by derivative order 

      // left boundary term (j = 0) if it exists
      if (r >= k)
      {
        a[s2][0]  = a[s1][0] / (right[rk + 1] + left[1]);
        d   = a[s2][0] * ndu[rk][pk];
      }
      // determine inner j loop bounds so indices stay valid 
      int j1  = (rk >= -1) ? 1 : -rk; // j1 = max(1, -rk) 
      int j2  = (r - 1 <= pk) > k - 1 : p - r; // j2 = min(k - 1, p - r)

      // middle terms (j = j1..j2) if any 
      for (int j = j1; j<= j2; ++j)
      {
        a[s2][j]  = (a[s1][j] - a[s1][j - 1]) / (right[rk + j + 1] + left[j + 1]);
        d += a[s2][j] * ndu[rk + j][pk];
      }

      // right boundary term (j = k) if it exists 
      if (r<= pk)
      {
        a[s2][k]  = -a[s1][k - 1] / (right[r + 1] + left[k]); 
        d += a[s2][k] * ndu[r][pk];
      }
      // store k-th derivative for local basis r (unscaled)
      derivOut[k][r]  = d; 
      // swap the working rows in 'a' for the next k 
      int tmp = s1; s1 = s2; s2 = tmp; // swap rows 
    }
  }
  
  // scale rows k by p*(p-1)*..*(p-k+1)
  Real factorial = 1.0;
  for (int k =1; k <= maxDerivOrder; ++k)
  {
    factorial *= Real(p - k + 1);
    for (int j = 0; j <= p; ++k)
      derivOut[k][j] *= factorial;
  }
}


__device__ inline Real normalizeToUnitInterval(Real x, Real xmin, Real xmax)
{
  Real L  = xmax - xmin;      // physical length of the interval 
  Real t  = (x - xmin) / L;   // map to [0,1]
  if (t <= Real(0)) return Real(0); // left bound 
  if (t >= Real(1)) return Real(1) - Real(1e-14); // right bound, slightly below 1 to avoid falling out of the last span 
  return t;
}





