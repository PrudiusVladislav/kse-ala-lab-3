
---

## Теоретична частина

### Які основні етапи включає SVD розклад ...

1. Сформувати матрицю (в даному випадку DataFrame з оцінками фільмів).

2. виконати SVD розклад матриці.
    - під час розкладу, треба ще обрати параметр k - к-сть найбільш важливих сингулярних значень, що залишаються.

3. зробити 'predictions' - відновити матрицю оцінок для прогнозування.
   - для цього перемножуємо матриці U, S, Vt.
   - в цій роботі ще враховуються особливості оцінки користувачів, тож до результату ще додається середня оцінка кожного користувача.  

#### ... і як цей метод можна застосувати до вирішення задачі підбору рекомендацій для певного користувача? (0.5 б.)

- можна сформувати 'профіль' користувача використовуючи матрицю \( U_k \).
  
- прогнозувати оцінки для всіх фільмів, які ще не були оцінені користувачем.

- ну і власне рекомендувати фільми, які мають найвищі прогнозовані оцінки.

### В яких сферах застосовується SVD? (0.5 б.)

1. аналіз текстів (LSA)
2. рекомендаційні системи
3. аналіз геномних даних (біоінформатика)

### Як вибір параметра k у SVD розкладі впливає на результат? (0.5 б.)

- Більший k - більше ресурсів використано, і якщо завеликий - забагато шуму. 
- Менший k - ефективніше, але може бути втрачена важлива інформація, якщо замале значення, 
однак менше k зазвичай дає точніші результати

### Які основні переваги та недоліки має SVD? (0.5 б.)

-  **Переваги**:
  - можливість зменшити розмірність даних, залишаючи найважливіші патерни.
  - можливість використати для будь-якої матриці.

-  **Недоліки**:
  - відносно 'дороге' обчислення
  - необхідність підготовки даних (наприклад, заповнення пропусків). але це впринципі для більшості методів

---