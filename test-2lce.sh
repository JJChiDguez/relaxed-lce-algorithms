#/bin/bash

echo "Running n=16 -> check logs/EXC_n16_q17.lce"
sage -python test_section4.1.py -n 16 -q 17 -b &> logs/EXC_n16_q17.lce

echo "Running n=16 -> check logs/EXC_n16_q7.lce"
sage -python test_section4.1.py -n 16 -q 7 -b &> logs/EXC_n16_q7.lce

echo "Running n=16 -> check logs/EXC_n16_q11.lce"
sage -python test_section4.1.py -n 16 -q 11 -b &> logs/EXC_n16_q11.lce

echo "Running n=16 -> check logs/EXC_n16_q31.lce"
sage -python test_section4.1.py -n 16 -q 31 -b &> logs/EXC_n16_q31.lce

echo "Running n=24 -> check logs/EXC_n24_q17.lce"
sage -python test_section4.1.py -n 24 -q 17 -b &> logs/EXC_n24_q17.lce

echo "Running n=24 -> check logs/EXC_n24_q7.lce"
sage -python test_section4.1.py -n 24 -q 7 -b &> logs/EXC_n24_q7.lce

echo "Running n=24 -> check logs/EXC_n24_q31.lce"
sage -python test_section4.1.py -n 24 -q 31 -b &> logs/EXC_n24_q31.lce

echo "Running n=32 -> check logs/EXC_n32_q17.lce"
sage -python test_section4.1.py -n 32 -q 17 -b &> logs/EXC_n32_q17.lce

echo "Running n=32 -> check logs/EXC_n32_q11.lce"
sage -python test_section4.1.py -n 32 -q 11 -b &> logs/EXC_n32_q11.lce

echo "Running n=32 -> check logs/EXC_n32_q7.lce"
sage -python test_section4.1.py -n 32 -q 7 -b &> logs/EXC_n32_q7.lce

echo "Running n=32 -> check logs/EXC_n32_q31.lce"
sage -python test_section4.1.py -n 32 -q 31 -b &> logs/EXC_n32_q31.lce

echo "Running n=40 -> check logs/EXC_n40_q17.lce"
sage -python test_section4.1.py -n 40 -q 17 -b &> logs/EXC_n40_q17.lce

echo "Running n=40 -> check logs/EXC_n40_q11.lce"
sage -python test_section4.1.py -n 40 -q 11 -b &> logs/EXC_n40_q11.lce

echo "Running n=40 -> check logs/EXC_n40_q7.lce"
sage -python test_section4.1.py -n 40 -q 7 -b &> logs/EXC_n40_q7.lce

echo "Running n=40 -> check logs/EXC_n40_q31.lce"
sage -python test_section4.1.py -n 40 -q 31 -b &> logs/EXC_n40_q31.lce

