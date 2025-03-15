# Бот, расчитывающий оценку стоимости недвижимости в зависимости от введенных пользователем данных

Для начала ознакомьтесь с официальным [туториалом](https://core.telegram.org/bots/tutorial), чтобы получить свой токен и URL для бота. Далее загрузите файлы данного репозитория себе на диск, создайте файл под названием `token.txt` и поместите в него свой токен. Создайте виртуальное окружение, установите зависимости из файла `requirements.txt` и запустите файл `tgbot.py`.

```
# копируем репозиторий себе на диск
git clone https://github.com/opavelkachalo/mos_real_estate_bot.git
# переходим в скопированную директорию
cd mos_real_estate_bot
# создаем виртуальное окружение
python3 -m venv .venv
# активируем виртуальное окружение
source .venv/bin/activate
# устанавливаем зависимости
pip install -r requirements.txt
# записываем токен в файл
echo "YOUR_TOKEN" > token.txt
# запускаем бота
python3 tgbot.py
```

Чтобы начать пользоваться ботом, перейдите по ссылке, которую вы получили при создании бота и дайте ему команду `/start`. Чтобы прекратить диалог, воспользуйтесь командой `/cancel`.

Данный бот был разработан в рамках проекта по научно-исследовательскому семинару «Информационная бизнес-аналитика».
