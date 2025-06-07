#include <iostream>
using namespace std;

class Node {
    int data;
public:
    Node *next;
    int getdata() {
        return data;
    }
    void setdata(int d) {
        data = d;
    }
   Node(int d) {
        data = d;
        next = nullptr;
    }
};

class Linkedlist {
    Node* head;
public:
    Linkedlist() {
        head = nullptr;
    }

    void insert(int data) {
        Node *temp = new Node(data);
        if(head == nullptr) {
            head = temp;
            return;
        }
        Node* curr = head;
        while(curr->next != nullptr)
            curr = curr->next;
        curr->next = temp;
    }

    void display() {
        Node* curr = head;
        while(curr != nullptr) {
            cout << curr->getdata() << " -> ";
            curr = curr->next;
        }
        cout << "NULL" << endl;
    }
};

int main() {
    Linkedlist L;
    L.insert(10);
    L.insert(20);
    L.insert(30);
    L.display();

    return 0;
}
