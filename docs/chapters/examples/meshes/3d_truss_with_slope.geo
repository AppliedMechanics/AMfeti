// Gmsh project created on Mon Nov 23 14:18:58 2020
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 0.25};
//+
Point(2) = {0, 1, 0, 0.25};
//+
Point(3) = {2, 0.25, 0, 0.25};
//+
Point(4) = {2, 0.75, 0, 0.25};
//+
Point(5) = {2, 0.75, 1, 0.25};
//+
Point(6) = {2, 1, 1, 0.25};
//+
Point(7) = {2, 1, 0, 0.25};
//+
Point(8) = {0, 1, 1, 0.25};
//+
Point(9) = {0, 0, 1, 0.25};
//+
Recursive Delete {
  Point{3}; 
}
//+
Line(1) = {2, 8};
//+
Line(2) = {8, 9};
//+
Line(3) = {9, 1};
//+
Line(4) = {1, 2};
//+
Line(5) = {2, 7};
//+
Line(6) = {7, 4};
//+
Line(7) = {4, 1};
//+
Line(8) = {9, 5};
//+
Line(9) = {5, 6};
//+
Line(10) = {6, 8};
//+
Line(11) = {4, 5};
//+
Line(12) = {7, 6};
//+
Line Loop(1) = {10, 2, 8, 9};
//+
Plane Surface(1) = {1};
//+
Line Loop(2) = {12, -9, -11, -6};
//+
Plane Surface(2) = {2};
//+
Line Loop(3) = {5, 6, 7, 4};
//+
Plane Surface(3) = {3};
//+
Line Loop(4) = {7, -3, 8, -11};
//+
Plane Surface(4) = {4};
//+
Line Loop(5) = {4, 1, 2, 3};
//+
Plane Surface(5) = {5};
//+
Line Loop(6) = {12, 10, -1, 5};
//+
Plane Surface(6) = {6};
//+
Surface Loop(1) = {6, 2, 1, 5, 3, 4};
//+
Volume(1) = {1};
//+
Physical Volume('material') = {1};
//+
//+
Physical Surface("neumann") = {6};
//+
Physical Surface("dirichlet") = {5};
