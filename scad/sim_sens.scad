union(){
    translate(v=[0.005, 0.014, 0.003]){
        rotate(a=[0, 90, 0]){
            color("orange"){
                cylinder(h=0.11, r=0.003, $fn=100);
            };
        };
    };
    translate(v=[0.005, 0.028, 0.003]){
        rotate(a=[0, 90, 0]){
            color("orange"){
                cylinder(h=0.11, r=0.003, $fn=100);
            };
        };
    };
    translate(v=[0.005, 0.042, 0.003]){
        rotate(a=[0, 90, 0]){
            color("orange"){
                cylinder(h=0.11, r=0.003, $fn=100);
            };
        };
    };
    translate(v=[0, 0, 0]){
        color("yellow",0.5){
            cube(size=[0.118, 0.0571, 0.00635]);
        };
    };
    translate(v=[0.001, 0.06381, 0.003]){
        color("orange"){
            cube(size=[0.11599999999999999, 0.055099999999999996, 0.0012699999999999999]);
        };
    };
    translate(v=[0, 0.06281, 0]){
        color("yellow",0.6){
            cube(size=[0.118, 0.0571, 0.00635]);
        };
    };
};
