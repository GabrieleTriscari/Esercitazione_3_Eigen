#include <iostream>
#include "Eigen/Eigen"
#include <vector>
#include <cmath>
using namespace std;
using namespace Eigen;

VectorXd SoluzionePALU(const MatrixXd& A, const VectorXd& b) {   // Risoluzione con il metodo PALU
    PartialPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

VectorXd SoluzioneQR(const MatrixXd& A, const VectorXd& b) {    // Risoluzione con il metodo QR
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

double ErroreRelativo(const MatrixXd& A, const VectorXd& x, const VectorXd& b) {  // Calcolo dell'errore relativo
    VectorXd residuo = (A * x) - b;
    return residuo.norm() / b.norm();
}

int main() {
    vector<MatrixXd> matriciA;      //Creazione di un vettore di matrici
    vector<VectorXd> vettoriB;       //Creazione di un vettore di vettori
    
    MatrixXd A1(2, 2); 
    A1 << 0.554701962252291, -0.03770900990025203,
            0.8320502943378437, -0.9992878623566787;
    matriciA.push_back(A1);
    
    MatrixXd A2(2, 2);
    A2 << 0.554701962252291, -0.5540673164667656,
            0.8320502943378437, -0.8324762492991315;
     matriciA.push_back(A2);
    
    MatrixXd A3(2, 2);
    A3 << 0.554701962252291, -0.5547019558519056,
            0.8320502943378437, -0.8320502947645361;
    matriciA.push_back(A3);
    
    Vector2d b1;
    b1 << -0.5169911863249772, 0.1672384680188350;
    vettoriB.push_back(b1);
    
    Vector2d b2;
    b2 << -0.0006394645785530173, 0.0004259549612877223;
    vettoriB.push_back(b2);
    
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433961e-10;
    vettoriB.push_back(b3);
    
    
    for (size_t i = 0; i < matriciA.size(); ++i) {
        cout << "Sistema " << i + 1 << ":\n";
    
        VectorXd x_palu = SoluzionePALU(matriciA[i], vettoriB[i]);
        VectorXd x_qr = SoluzioneQR(matriciA[i], vettoriB[i]);
    
        cout << "Soluzione con PALU: " << x_palu.transpose() << "\n";
        cout << "Errore relativo PALU: " << ErroreRelativo(matriciA[i], x_palu, vettoriB[i]) << "\n";
    
        cout << "Soluzione con QR:   " << x_qr.transpose() << "\n";
        cout << "Errore relativo QR: " << ErroreRelativo(matriciA[i], x_qr, vettoriB[i]) << "\n";
    
        }
    return 0;
}