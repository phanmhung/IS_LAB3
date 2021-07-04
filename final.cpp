
#include <iostream>
#include <ctime> 
#include <string.h> 
using namespace std;


string months[12]={ "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"};
string zodiak[13] = { "Козерог", "Водолей", "Рыбы", "Овен", "Телец", "Близнецы", "Рак", "Лев", "Дева", "Весы", "Скорпион", "Стрелец", "Козерог"};


double random_num()                  // рандомайзер 
{  
    int range = 101;
    double random_int = (1+rand() % range); 
    random_int /= 100;
    return random_int;
}


int main()
{
    setlocale(LC_ALL, "Russian");
    srand((unsigned)(time(0)));

    double alpha = 0.3, nu = 0.008;  // коэффициент нейрона в активационной функции alpha и скорость обучения ИНС nu   
    int amnet = 1;                   // количество нейронов скрытого слоя


    double*** net;                   // нейронная сеть
    net = new double** [2];          // с 2-мя слоями: скрытым и выходным

    net[0] = new double* [amnet];    // создание указателей на нейроны скрытого слоя 
    net[1] = new double* [1];        // создание указателей на нейроны выходного слоя 

    int i, j, k;                     // создание массива весов и смещений нейронов
    for (i = 0; i < amnet; i++)    
        net[0][i] = new double[3];   
    
    net[1][0] = new double[amnet + 1];


    double** f;                      // значения активационной функции нейронов
    f = new double* [2];
    f[0] = new double[2/*amnet*/];
    f[1] = new double[1];

    double* ff;                      // значения входных нейронов
    ff = new double[2];
  
    double** errnet;                 // ошибки нейронов
    errnet = new double* [2];
    errnet[0] = new double[amnet]; 
    errnet[1] = new double[1]; 
    

    for (i = 0; i < 1; i++)          // инициализация весов и смещений
        for (j = 0; j < amnet; j++)
            for (k = 0; k < 3; k++) 
                net[i][j][k] = random_num();
            

    for (k = 0; k < amnet + 1; k++) 
        net[1][0][k] = random_num();
     
  
    //Обучающая выборка
    int date[52]  = { 2, 9, 15, 19, 20, 27, 9, 18, 19, 25, 7, 20, 21, 27, 7, 20, 21, 27, 7, 20, 21, 27, 7, 20, 21, 27, 9, 22, 23, 29, 9, 22, 23, 29, 9, 22, 23, 29,  9, 22, 23, 29,  8, 21, 22, 28,  8, 21, 22, 24, 27, 30 };
    int month[52] = { 1, 1,  1,  1,  1,  1, 2,  2,  2,  2, 3,  3,  3,  3, 4,  4,  4,  4, 5,  5,  5,  5, 6,  6,  6,  6, 7,  7,  7,  7, 8,  8,  8,  8, 9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12 };
    int ans[52]   = { 1, 1,  1,  1,  2,  2, 2,  2,  3,  3, 3,  3,  4,  4, 4,  4,  5,  5, 5,  5,  6,  6, 6,  6,  7,  7, 7,  7,  8,  8, 8,  8,  9,  9, 9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13 }; 

   
    int c, count = 0, epoxa = 0;    
    double err;


    cout << "Start INS Learning!" << endl;              //обучение ИНС

    for (epoxa = 0; epoxa < 175; epoxa++)               //эпоха обучения
        for (c = 0; c < 52; c++)                        //ИНС обучается полседовательно каждому набору из обучающей выборки
        {          
            err = 1;
            count = 0;

            while (abs(err) > 0.1 && count < 20000)     //добиваемся нужной точности работы ИНС при ограниченном числе итераций обучения для одного набора
            {

                f[0][0] = month[c];
                f[0][1] = date[c];

                ff[0] = month[c];
                ff[1] = date[c];

                //запуск работы нейронной сети

                for (i = 0; i < 1; i++)
                    for (j = 0; j < amnet; j++)                      
                        f[i][j] = f[i][0] * net[i][j][1] + alpha * f[i][1] * net[i][j][2] + net[i][j][0];
                      

                f[1][0] = f[0][0] * net[1][0][1] + net[1][0][0];
               
                //ошибка сети
                err = f[1][0] - ans[c];      

                //ошибки нейронов
                errnet[1][0] = err;                         
                errnet[0][0] = net[1][0][1] * errnet[1][0];
              

                //коррекция весов 
                for (i = 0; i < 1; i++)
                    for (j = 0; j < amnet; j++)
                        for (k = 1; k < 3; k++)
                        {
                            if (k == 1)
                                net[i][j][k] = net[i][j][k] - nu * errnet[0][j] * ff[k - 1];
                            else  net[i][j][k] = net[i][j][k] - alpha * nu * errnet[0][j] * ff[k - 1];                           
                        }

                for (k = 1; k < amnet + 1; k++)
                    net[1][0][k] = net[1][0][k] - nu * errnet[1][0] * f[0][k - 1];


                //коррекция смещений
                for (i = 0; i < 1; i++)
                    for (j = 0; j < amnet; j++)
                        net[i][j][0] = net[i][j][0] - nu * errnet[0][j];

                net[1][0][0] = net[1][0][0] - nu * errnet[1][0];
                
                count++;
            }
        }

    cout << "INS Learning finished!" << endl << endl;


    //Тестовая выборка
    int testdate[24]   = {12, 29, 11, 28, 10, 27, 9, 26, 8, 25, 7, 24, 6, 23, 5, 22, 4, 21,  3,  20,  2, 19,  1, 30};
    int testmonth[24]  = { 1,  1,  2,  2,  3,  3, 4,  4, 5,  5, 6,  6, 7,  7, 8,  8, 9,  9, 10,  10, 11, 11, 12, 12};
    double testans[24] = { 1,  2,  2,  3,  3,  4, 4,  5, 5,  6, 6,  7, 7,  8, 8,  8, 9,  9, 10,  10, 11, 11, 12, 13};


    int prepos = 0, help1 = 0;
    double correct = 0, correctness = 0, help = 0;


    //запуск работы нейронной сети

    for (count = 0; count < 24; count++) {

        f[0][0] = testmonth[count];
        f[0][1] = testdate[count];

        cout << "Дата рождения: " << f[0][1] << " " << months[testmonth[count]-1] <<endl;
       
        for (i = 0; i < 1; i++)
            for (j = 0; j < amnet; j++)           
                f[i][j] = f[i][0] * net[i][j][1] + alpha * f[i][1] * net[i][j][2] + net[i][j][0];
            
        f[1][0] = f[0][0] * net[1][0][1] + net[1][0][0];


        //обработка полученного нейронной сетью ответа

        prepos = f[1][0];
        help = prepos;

        if (f[1][0] - help >= 0.5) help++;

        help1 = help;

        cout << "Нейронная сеть предполагает, что Вы: " << zodiak[help1 - 1] << endl;

        help1 = testans[count];

        cout << "В действительности Ваш знак зодиака: " << zodiak[help1 - 1] << endl;

        cout << "Результат предсказания: ";

        if (help == testans[count]) { cout << "Success" << endl; correct++; }
        else cout << "Failed" << endl;
      
        cout << endl;
    }

    correctness = correct / count * 100;
    cout << endl << "Точность работы искусственной нейронной сети: " << correctness << " процента" << endl;
   // system("pause"); // Команда задержки экрана


   // cout << endl;
    int yourmonth, yourdate, enough;

    cout << endl<<endl << "Если хотите получить ещё одно предсказание на знак зодиака, введите 1. В противном случае, введите 0." << endl;
    cin >> enough;

    while (enough == 1) {

        cout << endl<< "Введите день своего рождения: ";
        cin >> yourdate;
        cout << "Введите месяц своего рождения: ";
        cin >> yourmonth;



        f[0][0] = yourmonth;
        f[0][1] = yourdate;

        cout << "Дата рождения: " << f[0][1] << " " << months[yourmonth - 1] << endl;

        for (i = 0; i < 1; i++)
            for (j = 0; j < amnet; j++)
                f[i][j] = f[i][0] * net[i][j][1] + alpha * f[i][1] * net[i][j][2] + net[i][j][0];

        f[1][0] = f[0][0] * net[1][0][1] + net[1][0][0];


        //обработка полученного нейронной сетью ответа

        prepos = f[1][0];
        help = prepos;

        if (f[1][0] - help >= 0.5) help++;

        help1 = help;

        cout << "Нейронная сеть предполагает, что Вы: " << zodiak[help1 - 1] << endl;
        cout << endl<<"Если хотите получить ещё одно предсказание на знак зодиака, введите 1. В противном случае, введите 0."<<endl;

        cin >> enough;
    }
}
