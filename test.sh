 #!/bin/sh
python3 main.py --base-dir seoul_test3/ma2c train --config-dir config/config_ma2c_seoul.ini --test-mode no_test
sleep 10
python3 main.py --base-dir seoul_test3/iqll train --config-dir config/config_iqll_seoul.ini --test-mode no_test
sleep 10
# python3 main.py --base-dir seoul_test2/ia2c train --config-dir config/config_ia2c_seoul.ini --test-mode no_test
# sleep 10 

