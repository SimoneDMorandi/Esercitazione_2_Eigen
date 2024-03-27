#include <iostream>
#include <Eigen/Eigen>
#include <limits>
#include <iomanip>

using namespace std;
using namespace Eigen;

// Definisco la funzione che controlla se la matrice è singolare
bool check_singularity( MatrixXd A){
    double epsilon = std::numeric_limits<double>::epsilon();
    if (abs(A.determinant()) < epsilon) {
        cerr << "Matrix is singular" << endl;
        return 1;
    }
    return 0;
}
// Definisco la funzione che calcola la fattorizzazione PA=LU e risolve il sistema lineare Ax=b
VectorXd PALU(MatrixXd A, VectorXd b)
{
    PartialPivLU<Eigen::MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

// Definisco la funzione che calcola la fattorizzazione QR e risolve il sistema lineare Ax=b
VectorXd QR(MatrixXd A, VectorXd b)
{
    HouseholderQR<Eigen::MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}

int main()
{
    // Definisco la dimension comune a tutti i sistemi
    unsigned int n = 2;

    // Definisco il vettore soluzione
    VectorXd x_0(n);
    x_0 = - x_0.setOnes();

    // Imposto la formattazione per l'output
    cout << fixed << scientific << setprecision(2);

    // Primo sistema
    MatrixXd A1 = MatrixXd::Zero(n, n);
    VectorXd b1(n);
    b1.setZero();
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    bool singular = check_singularity(A1);
    if (singular == 0) {
        VectorXd x1_palu = PALU(A1, b1);
        VectorXd x1_qr = QR(A1, b1);
        double errrel_1_palu = (x1_palu - x_0).norm() / x_0.norm();
        double errrel_1_qr = (x1_qr - x_0).norm() / x_0.norm();
        cout << "Soluzione del primo sistema con PALU: " << endl << x1_palu << endl << endl;
        cout << "Soluzione del primo sistema con QR: " << endl << x1_qr << endl << endl;
        cout << "Errore relativo per il primo sistema con PALU: " << errrel_1_palu << endl;
        cout << "Errore relativo per il primo sistema con QR: " << errrel_1_qr << endl << endl;
    }

    // Secondo sistema
    MatrixXd A2 = MatrixXd::Zero(n, n);
    VectorXd b2(n);
    b2.setZero();
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    singular = check_singularity(A2);
    if (singular == 0) {
        VectorXd x2_palu = PALU(A2, b2);
        VectorXd x2_qr = QR(A2, b2);
        double errrel_2_palu = (x2_palu - x_0).norm() / x_0.norm();
        double errrel_2_qr = (x2_qr - x_0).norm() / x_0.norm();
        cout << "Soluzione del secondo sistema con PALU: " << endl << x2_palu << endl << endl;
        cout << "Soluzione del secondo sistema con QR: " << endl << x2_qr << endl << endl;
        cout << "Errore relativo per il secondo sistema con PALU: " << errrel_2_palu << endl;
        cout << "Errore relativo per il secondo sistema con QR: " << errrel_2_qr << endl << endl;
    }

    // Terzo sistema
    MatrixXd A3 = MatrixXd::Zero(n, n);
    VectorXd b3(n);
    b3.setZero();
    A3 <<5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    singular = check_singularity(A3);
    if (singular == 0) {
        VectorXd x3_palu = PALU(A3, b3);
        VectorXd x3_qr = QR(A3, b3);
        double errrel_3_palu = (x3_palu - x_0).norm() / x_0.norm();
        double errrel_3_qr = (x3_qr - x_0).norm() / x_0.norm();
        cout << "Soluzione del terzo sistema con PALU: " << endl << x3_palu << endl << endl;
        cout << "Soluzione del terzo sistema con QR: " << endl << x3_qr << endl << endl;
        cout << "Errore relativo per il terzo sistema con PALU: " << errrel_3_palu << endl;
        cout << "Errore relativo per il terzo sistema con QR: " << errrel_3_qr << endl << endl;
    }

    return 0;
}
