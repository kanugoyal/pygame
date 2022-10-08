from operator import truediv


board = ['_','_','_','_','_','_','_','_','_']

def display_board():
    print('|' + board[0] + '|' + board[1] + '|' + board[2] + '|')
    print('|' + board[3] + '|' + board[4] + '|' + board[5] + '|')
    print('|' + board[6] + '|' + board[7] + '|' + board[8] + '|')


def check_winner(board):
    p1 = 'x'
    p2 = 'o'
    if board[0] == board[1] == board[2] == p1 or board[0] == board[1] == board[2] == p2:
        return True
    elif board[3] == board[4] == board[5] == p1 or board[3] == board[4] == board[5] == p2:
        return True
    elif board[6] == board[7] == board[8] == p1 or board[6] == board[7] == board[8] == p2:
        return True
    elif board[0] == board[3] == board[6] == p1 or board[0] == board[3] == board[6] == p2:
        return True
    elif board[1] == board[4] == board[7] == p1 or board[1] == board[4] == board[7] == p2:
        return True
    elif board[2] == board[5] == board[8] == p1 or board[2] == board[5] == board[8] == p2:
        return True
    elif board[0] == board[4] == board[8] == p1 or board[0] == board[4] == board[8] == p2:
        return True
    elif board[2] == board[4] == board[6] == p1 or board[2] == board[4] == board[6] == p2:
        return True
    else:
        return False


def input_value(board):
    x = int(input("Enter your position (1-9): "))
    if board[x-1] != '_':
        print("Position filled, please enter new value :")
        return input_value(board)

    else:
        return x


player1 = input("Enter Player name 1: ")
player2 = input("Enter Player name 2: ")
display_board()
for i in range(9):
    if i % 2 == 0:
        x = input_value(board)
        board[x-1] = 'x'
        display_board()
        if check_winner(board):
            print(f"player 1 {player1} wins")
            break
    else:
        x = input_value(board)
        board[x-1] = 'o'
        display_board()
        if check_winner(board):
            print(f"player 2 {player2} wins")
            break
print("Game Over")