from lcapy import R

Rtot = (R(1) + (R(2) + R(2)| R(3) + (R(4)|R(5)+R(5)+R(5)+R(5))))

print(Rtot.simplify())
Rtot.draw('test12.pdf')