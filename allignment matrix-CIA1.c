#include <iostream>
#include<stdio.h>
#include<string.h>
using namespace std;

int check_matrix(int i,int j,int le,int *arr,string str1,string str2)
{
    int value = 0;
    if(str1[i-1] == str2[j-1])
    {
        value = (*(arr+i-1)+j-1);
        value = value + 2;
        return value;
    }
    else
    {
        return 0;
    }
}

int main() 
{
    string str1,str2;
    int le;
    cout <<"Enter String 1 : ";
    //cin >> str1;
    str1 = "acacacgac";
    str2 = "atcacacac";
    cout << str1 <<"\n";
    cout <<"Enter String 2 : ";
    //cin >> str2;
    cout << str2 <<"\n";
    if(str1.length() != str2.length())
    {
        cout <<"Strings are not equal strength";
        return -1;
    }
    else
    {
        le = str1.length();
        int arr[le+1][le+1];

        for(int i = 0;i < le+1;i++)
        {
            for(int j = 0;j < le+1;j++)
            {
                arr[i][j] = i;
            }
        }

        for(int i = 0;i < le+1;i++)
        {
            for(int j = 0;j < le+1;j++)
            {
                if(i == 0 || j == 0)
                {
                    arr[i][j] = 0;
                }    
                else
                {
                     arr[i][j] = check_matrix(i,j,le,(int *)arr,str1,str2);
                    //print_mat(le,arr);
                    
                }
            }
        }
        cout << "\n";
        cout <<"  - ";
        for(int i = 0;i < le;i++)
        {
            cout << str1[i]<<" ";
        }
        cout << "\n";
        for(int i = 0;i < le+1;i++)
        {
            if(i == 0)
            {
                cout << "-";
            }
            cout << str2[i-1] << " ";
            for(int j = 0;j < le+1;j++)
            {
                cout <<arr[i][j]<< " ";
            }
            cout << "\n";
        }
    }
    return 0;
}