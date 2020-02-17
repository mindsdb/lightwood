pkill -9 visdom;

pip3 install -r requirements_dev.txt --user || pip install -r requirements_dev.txt --user || pip3 install -r requirements_dev.txt || pip install -r requirements_dev.txt;

visdom 2>&1 > /dev/null &

echo -ne '\n';
sleep 2s && echo -ne '\n';
sleep 2s && echo -ne '\n';

open http://localhost:8097 || xdg-open http://localhost:8097;
