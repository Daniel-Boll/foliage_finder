#!/bin/bash

wget -r --no-parent --spider --no-check-certificate "https://www.inf.unioeste.br/~adair/PID/Trabalhos/Imagens/" 2>&1 | grep '^--' | rg ".bmp" | awk '{print $3}' > urls.txt
aria2c -i ./urls.txt -x 16 -s 16 -j 16 --file-allocation=none --dir=../assets
