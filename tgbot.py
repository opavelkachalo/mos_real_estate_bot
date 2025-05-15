#! .venv/bin/python3
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
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

OBJECTS_PATH = 'pickles/'
with open("token.txt", "r") as token_file:
    TOKEN = token_file.readline()[:-1]

logging.basicConfig(
        filename='tgbot.log',
        encoding='utf-8',
        format='[%(asctime)s] %(message)s',
        level=logging.INFO
)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.info('Начало работы бота')

DATA = pd.DataFrame()
IS_NEW, MINUTES, METRO, DISTRICT, ROOMS, AREA, LIVING_AREA, KIT_AREA, FLOOR,\
        NUM_OF_FLOORS, WALL_MATERIAL, RENOVATION = range(12)


def make_dummy(df, column, cols):
    df = pd.get_dummies(df, columns=[column], dtype=int)
    for col in cols:
        if col not in df.columns:
            df[col] = [0]
    return df


def count_price_level(district):
    match district:
        case 'ЦАО' | 'ЗАО' | 'ЮЗАО' | 'САО' | 'СЗАО':
            price_level = 'high'
        case 'ЮАО' | 'СВАО' | 'ВАО':
            price_level = 'medium'
        case 'ЮВАО' | 'Новая Москва':
            price_level = 'low'
    return price_level


def load_objects(price_level):
    match price_level:
        case 'high':
            encoder_fname = OBJECTS_PATH + 'label_encoder_high.dat'
            scaler_fname = OBJECTS_PATH + 'scaler_high.dat'
            model_fname = OBJECTS_PATH + 'model_high.dat'
        case 'medium':
            encoder_fname = OBJECTS_PATH + 'label_encoder_medium.dat'
            scaler_fname = OBJECTS_PATH + 'scaler_medium.dat'
            model_fname = OBJECTS_PATH + 'model_mid.dat'
        case 'low':
            encoder_fname = OBJECTS_PATH + 'label_encoder_low.dat'
            scaler_fname = OBJECTS_PATH + 'scaler_low.dat'
            model_fname = OBJECTS_PATH + 'model_low.dat'
    with open(encoder_fname, 'rb') as encoder_f:
        encoder = pickle.load(encoder_f)
    with open(scaler_fname, 'rb') as scaler_f:
        scaler = pickle.load(scaler_f)
    with open(model_fname, 'rb') as model_f:
        model = pickle.load(model_f)
    return encoder, scaler, model


def drop_extra_cols(df, price_level):
    if price_level == 'high':
        df = df[['Минут до метро', 'Количество комнат',
                'Площадь', 'Жилая площадь', 'Кухня площадь',
                'Этаж', 'Количество этажей',
                'Станция метро (очищенная)', 'Тип_квартиры_код',
                'Материал стен_Блочный', 'Материал стен_Иные',
                'Материал стен_Кирпичный', 'Материал стен_Монолитно-кирпичный',
                'Материал стен_Монолитный', 'Материал стен_Панельный',
                'Ремонт_Без ремонта', 'Ремонт_Дизайнерский',
                'Ремонт_Евроремонт', 'Ремонт_Косметический', 'Ремонт_Неизвестно',
                'Округ_ЗАО', 'Округ_САО', 'Округ_СЗАО', 'Округ_ЦАО', 'Округ_ЮЗАО']]
    elif price_level == 'medium':
        df = df[['Минут до метро', 'Количество комнат',
                'Площадь', 'Жилая площадь', 'Кухня площадь',
                'Этаж', 'Количество этажей',
                'Станция метро (очищенная)', 'Тип_квартиры_код',
                'Материал стен_Блочный', 'Материал стен_Иные',
                'Материал стен_Кирпичный', 'Материал стен_Монолитно-кирпичный',
                'Материал стен_Монолитный', 'Материал стен_Панельный',
                'Ремонт_Без ремонта', 'Ремонт_Дизайнерский',
                'Ремонт_Евроремонт', 'Ремонт_Косметический', 'Ремонт_Неизвестно',
                'Округ_ВАО', 'Округ_СВАО', 'Округ_ЮАО']]
    else:
        df = df[['Минут до метро', 'Количество комнат',
                'Площадь', 'Жилая площадь', 'Кухня площадь',
                'Этаж', 'Количество этажей',
                'Станция метро (очищенная)', 'Тип_квартиры_код',
                'Материал стен_Блочный', 'Материал стен_Иные',
                'Материал стен_Кирпичный', 'Материал стен_Монолитно-кирпичный',
                'Материал стен_Монолитный', 'Материал стен_Панельный',
                'Ремонт_Без ремонта', 'Ремонт_Дизайнерский',
                'Ремонт_Евроремонт', 'Ремонт_Косметический', 'Ремонт_Неизвестно',
                'Округ_Новая Москва', 'Округ_ЮВАО']]
    return df


def label_encode(df, col, encoder, price_level):
    try:
        df[col] = encoder.transform(df[col])
    except ValueError:
        encoder.classes_ = np.append(encoder.classes_, df[col])
        f = open(f'{OBJECTS_PATH}label_encoder_{price_level}.dat', 'wb')
        pickle.dump(encoder, f)
        f.close()
        df[col] = encoder.transform(df[col])
    return df


def predict_price(df):
    price_level = count_price_level(df.loc[0, "Округ"])
    df = make_dummy(df, column='Материал стен', cols=[
        'Материал стен_Блочный', 'Материал стен_Иные',
        'Материал стен_Кирпичный',
        'Материал стен_Монолитно-кирпичный',
        'Материал стен_Монолитный',
        'Материал стен_Панельный'
    ])
    df = make_dummy(df, column='Ремонт', cols=[
        'Ремонт_Без ремонта', 'Ремонт_Дизайнерский',
        'Ремонт_Евроремонт', 'Ремонт_Косметический',
        'Ремонт_Неизвестно'
    ])
    df = make_dummy(df, column='Округ', cols=[
        'Округ_ВАО', 'Округ_ЗАО', 'Округ_Новая Москва',
        'Округ_САО', 'Округ_СВАО', 'Округ_СЗАО',
        'Округ_ЦАО', 'Округ_ЮАО', 'Округ_ЮВАО', 'Округ_ЮЗАО'
    ])
    df = drop_extra_cols(df, price_level)
    encoder, scaler, model = load_objects(price_level)
    df = label_encode(df, 'Станция метро (очищенная)', encoder, price_level)
    df[df.columns] = scaler.transform(df[df.columns])
    df = df.reindex(sorted(df.columns), axis=1)
    y = model.predict(df)
    return y[0]


async def start(update, context):
    reply_keyboard = [["Новостройка", "Вторичное"]]
    DATA = pd.DataFrame()
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
    user = update.message.from_user
    logging.info('Квартира пользователя %s: %s', user, ans)
    DATA['Тип_квартиры_код'] = [1 if ans == 'Новостройка' else 0]
    await update.message.reply_text(
        "Сколько времени в минутах занимает путь пешком до ближайшей станции метро?",
        reply_markup=ReplyKeyboardRemove(),
    )

    return MINUTES


async def minutes(update, context):
    DATA['Минут до метро'] = [float(update.message.text)]
    user = update.message.from_user
    logging.info('Минут до метро у пользователя %s: %s', user, update.message.text)
    await update.message.reply_text(
        "Введите название ближайшей станции метро:",
        reply_markup=ReplyKeyboardRemove(),
    )
    return METRO


async def metro(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Ближайшая станция метро пользователя %s: %s', user, ans)
    DATA['Станция метро (очищенная)'] = [update.message.text]
    reply_keyboard = [["ЮВАО", "ЮАО", "ЦАО", "ЗАО", "САО", "СЗАО", "ЮЗАО", "СВАО", "ВАО", "Новая Москва"]]
    await update.message.reply_text("В каком районе Москвы находится Ваша квартира?",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True
            )
    )
    return DISTRICT


async def district(update, context):
    DATA['Округ'] = [update.message.text]
    user = update.message.from_user
    logging.info('Район пользователя %s: %s', user, update.message.text)
    await update.message.reply_text(
        "Сколько комнат в Вашей квартире?",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ROOMS


async def rooms(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Комнат в квартире пользователя %s: %s', user, ans)
    DATA['Количество комнат'] = [float(ans)]
    await update.message.reply_text(
        "Какова общая площадь Вашей квартиры в квадратных метрах?",
        reply_markup=ReplyKeyboardRemove(),
    )
    return AREA


async def area(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Площадь квартиры пользователя %s: %s', user, ans)
    DATA['Площадь'] = [float(ans)]
    await update.message.reply_text(
        "Какова жилая площадь Вашей квартиры в квадратных метрах?"
    )
    return LIVING_AREA


async def living_area(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Жилая площадь квартиры пользователя %s: %s', user, ans)
    DATA['Жилая площадь'] = [float(ans)]
    await update.message.reply_text(
        "Какова площадь кухни в Вашей квартире в квадратных метрах?"
    )
    return KIT_AREA


async def kit_area(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Площадь кухни пользователя %s: %s', user, ans)
    DATA['Кухня площадь'] = [float(ans)]
    await update.message.reply_text(
        "На каком этаже расположена Ваша квартира?"
    )
    return FLOOR


async def floor(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Этаж пользователя %s: %s', user, ans)
    DATA['Этаж'] = [float(ans)]
    await update.message.reply_text(
        "Сколько всего этажей в Вашем доме?"
    )
    return NUM_OF_FLOORS


async def num_of_floors(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Кол-во этажей пользователя %s: %s', user, ans)
    DATA['Количество этажей'] = [float(ans)]
    reply_keyboard = [["Иные", "Кирпичный", "Монолитно-кирпичный", "Монолитный", "Панельный"]]
    await update.message.reply_text(
        "Какой материал стен в Вашей квартире?",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True
        )
    )
    return WALL_MATERIAL


async def wall_material(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Материал стен в квартире пользователя %s: %s', user, ans)
    DATA['Материал стен'] = [ans]
    reply_keyboard = [["Без ремонта", "Дизайнерский", "Евроремонт", "Косметический", "Неизвестно"]]
    await update.message.reply_text(
        "Какой ремонт в Вашей квартире?",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True
        )
    )
    return RENOVATION


async def renovation(update, context):
    ans = update.message.text
    user = update.message.from_user
    logging.info('Ремонт у пользователя %s: %s', user, ans)
    DATA['Ремонт'] = [ans]
    price = round(predict_price(DATA))
    await update.message.reply_text(f"Ваша квартира оценивается в {price:,} рублей")
    logging.info('Стоимость квартиры пользователя %s: %s', user, price)
    return ConversationHandler.END


async def cancel(update, context):
    user = update.message.from_user
    await update.message.reply_text("Увидимся в другой раз!")
    logging.info('Пользователь %s отменил диалог', user)
    return ConversationHandler.END


def main():
    application = Application.builder().token(TOKEN).build()

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            IS_NEW: [MessageHandler(filters.Regex("^(Новостройка|Вторичное)$"), is_new)],
            MINUTES: [MessageHandler(filters.TEXT & ~filters.COMMAND, minutes)],
            METRO: [MessageHandler(filters.TEXT & ~filters.COMMAND, metro)],
            DISTRICT: [MessageHandler(filters.Regex("^(ЮВАО|ЮАО|ЦАО|ЗАО|САО|СЗАО|ЮЗАО|СВАО|ВАО|Новая Москва)$"), district)],
            ROOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, rooms)],
            AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, area)],
            LIVING_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, living_area)],
            KIT_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, kit_area)],
            FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, floor)],
            NUM_OF_FLOORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, num_of_floors)],
            WALL_MATERIAL: [MessageHandler(filters.Regex("^(Иные|Кирпичный|Монолитно-кирпичный|Монолитный|Панельный)$"), wall_material)],
            RENOVATION: [MessageHandler(filters.Regex("^(Без ремонта|Дизайнерский|Евроремонт|Косметический|Неизвестно)$"), renovation)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
