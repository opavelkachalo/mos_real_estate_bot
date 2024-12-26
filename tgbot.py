#! .venv/bin/python3
import pickle
import lightgbm
from lightgbm import LGBMRegressor
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    PicklePersistence,
    filters,
)

with open("token.txt", "r") as token_file:
    TOKEN = token_file.readline()[:-1]

DATA = {}
IS_NEW, MINUTES, IS_MOSCOW, DISTRICT, ROOMS, AREA, KIT_AREA, FLOOR,\
        NUM_OF_FLOORS, RENOVATION = range(10)


def predict_price(values):
  # закодируем ремонт
  if values[0][8] == 'Cosmetic':
    values[0] += [0, 0, 0]
  elif values[0][8] == 'Designer':
    values[0] += [1, 0, 0]
  elif values[0][8] == 'European-style renovation':
    values[0] += [0, 1, 0]
  else:
    values[0] += [0, 0, 1]
  # удалим старое значение ремонта
  values[0].pop(8)

  # поделим на регион и москву
  # проверка на регион
  if values[0][2] == 0:
     # удаляем район и индикатор москвы, тк они дальше не нужны
    values[0].pop(8)
    values[0].pop(2)
    # загрузим модель для предсказания
    load_model = pickle.load(open('lightgbm_region', 'rb'))

  else:
    # удалим индикатор москвы, тк он дальше не нужен
    values[0].pop(2)
    # проверка на moscow_high
    if values[0][7] in ['zao', 'cao']:
      # кодируем район
      if values[0][7] == 'zao':
        values[0] += [1]
      else:
        values[0] += [0]
      # удалим старое значение района
      values[0].pop(7)
      # загрузим модель для предсказания
      load_model = pickle.load(open('lightgbm_high', 'rb'))

    # проверка на moscow_medium
    elif values[0][7] in ['sao', 'szao', 'uao', 'uzao']:
      # кодируем район
      if values[0][7] == 'sao':
        values[0] += [0, 0, 0]
      elif values[0][7] == 'szao':
        values[0] += [1, 0, 0]
      elif values[0][7] == ' uao':
        values[0] += [0, 1, 0]
      else:
        values[0] += [0, 0, 1]
      # удалим старое значение района
      values[0].pop(7)
      # загрузим модель для предсказания
      load_model = pickle.load(open('lightgbm_medium', 'rb'))

    # проверка на moscow_low
    else:
      # кодируем район
      if values[0][7] == 'nmao':
        values[0] += [0, 0, 0]
      elif values[0][7] == 'uvao':
        values[0] += [0, 1, 0]
      elif values[0][7] == 'vao':
        values[0] += [0, 0, 1]
      else:
        values[0] += [1, 0, 0]
      # удалим старое значение района
      values[0].pop(7)
      # загрузим модель для предсказания
      load_model = pickle.load(open('lightgbm_low', 'rb'))

  # делаем предсказание
  y_pred = load_model.predict(values)
  return y_pred[0]


async def start(update, context):
    reply_keyboard = [["Новая", "Не Новая"]]
    DATA = {}
    await update.message.reply_text(
            "Здравствуйте! Для оценки квартиры, я бы"
            "хотел задать вам несколько вопросов."
            "Вы можете отменить наш диалог в любой момент, написав /cancel"
            "\n\nВаша квартира новая или нет?",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True
            )
    )

    return IS_NEW


async def is_new(update, context):
    ans = update.message.text
    DATA['is_new'] = 1 if ans == 'Новая' else 0
    await update.message.reply_text(
        "Сколько времени в минутах занимает путь пешком до ближайшей станции метро?",
        reply_markup=ReplyKeyboardRemove(),
    )

    return MINUTES


async def minutes(update, context):
    DATA['minutes'] = float(update.message.text)
    reply_keyboard = [["В Подмосковье", "В Москве"]]
    await update.message.reply_text("Ваша квартира находится в Москве или в Подмосковье?",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True
            )
    )
    return IS_MOSCOW


async def is_moscow(update, context):
    ans = update.message.text
    if ans == 'В Москве':
        DATA['is_moscow'] = 1
        reply_keyboard = [["uvao", "uao", "cao", "zao", "sao", "szao", "uzao", "svao", "vao", "nmao"]]
        await update.message.reply_text("В каком районе Москвы находится Ваша квартира?",
                reply_markup=ReplyKeyboardMarkup(
                    reply_keyboard, one_time_keyboard=True
                )
        )
        return DISTRICT
    else:
        DATA['is_moscow'] = 0
        DATA['district'] = ''
        await update.message.reply_text(
            "Сколько комнат в Вашей квартире?",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ROOMS


async def district(update, context):
    DATA['district'] = update.message.text
    await update.message.reply_text(
        "Сколько комнат в Вашей квартире?",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ROOMS


async def rooms(update, context):
    ans = update.message.text
    DATA['rooms'] = float(ans)
    await update.message.reply_text(
        "Какова общая площадь Вашей квартиры в квадратных метрах?",
        reply_markup=ReplyKeyboardRemove(),
    )
    return AREA


async def area(update, context):
    ans = update.message.text
    DATA['area'] = float(ans)
    await update.message.reply_text(
        "Какова площадь кухни в Вашей квартире в квадратных метрах?"
    )
    return KIT_AREA


async def kit_area(update, context):
    ans = update.message.text
    DATA['kit_area'] = float(ans)
    await update.message.reply_text(
        "На каком этаже расположена Ваша квартира?"
    )
    return FLOOR


async def floor(update, context):
    ans = update.message.text
    DATA['floor'] = float(ans)
    await update.message.reply_text(
        "Сколько всего этажей в Вашем доме?"
    )
    return NUM_OF_FLOORS


async def num_of_floors(update, context):
    ans = update.message.text
    DATA['num_of_floors'] = int(ans)
    reply_keyboard = [["Cosmetic", "European-style renovation", "Designer", "Without renovation"]]
    await update.message.reply_text(
        "Какой ремонт в Вашей квартире?",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True
        )
    )
    return RENOVATION


async def renovation(update, context):
    ans = update.message.text
    DATA['renovation'] = ans
    features = [[
        DATA['is_new'],
        DATA['minutes'],
        DATA['is_moscow'],
        DATA['rooms'],
        DATA['area'],
        DATA['kit_area'],
        DATA['floor'],
        DATA['num_of_floors'],
        DATA['renovation'],
        DATA['district']
    ]]
    price = round(predict_price(features))
    # await update.message.reply_text(f"{features}")
    await update.message.reply_text(f"Ваша квартира оценивается в {price:,} рублей")
    return ConversationHandler.END


async def cancel(update, context):
    await update.message.reply_text("Увидимся в другой раз!")
    return ConversationHandler.END


def main():
    application = Application.builder().token(TOKEN).build()

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            IS_NEW: [MessageHandler(filters.Regex("^(Новая|Не Новая)$"), is_new)],
            MINUTES: [MessageHandler(filters.TEXT & ~filters.COMMAND, minutes)],
            IS_MOSCOW: [MessageHandler(filters.Regex("^(В Подмосковье|В Москве)$"), is_moscow)],
            DISTRICT: [MessageHandler(filters.Regex("^(uvao|uao|cao|zao|sao|szao|uzao|svao|vao|nmao)$"), district)],
            ROOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, rooms)],
            AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, area)],
            KIT_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, kit_area)],
            FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, floor)],
            NUM_OF_FLOORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, num_of_floors)],
            RENOVATION: [MessageHandler(filters.Regex("^(Cosmetic|European-style renovation|Designer|Without renovation)$"), renovation)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
