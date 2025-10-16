import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 실제 향수 데이터 (브랜드, 이름, 년도, 성별, 스타일, 탑노트, 미들노트, 베이스노트)
REAL_PERFUMES = [
    # Chanel
    ("Chanel", "No 5", 1921, "Women", "floral",
        {"Neroli": 10, "Bergamot": 10, "Lemon": 10},
        {"Jasmine": 15, "Rose": 15, "Ylang-Ylang": 10},
        {"Vanilla": 12, "Sandalwood": 10, "Vetiver": 8}),
    ("Chanel", "Coco Mademoiselle", 2001, "Women", "oriental",
        {"Orange": 10, "Bergamot": 10, "Grapefruit": 10},
        {"Rose": 15, "Jasmine": 15, "Lychee": 10},
        {"Patchouli": 10, "Vanilla": 12, "Musk": 8}),
    ("Chanel", "Bleu de Chanel", 2010, "Men", "woody",
        {"Grapefruit": 10, "Lemon": 10, "Mint": 10},
        {"Ginger": 13, "Nutmeg": 14, "Jasmine": 13},
        {"Sandalwood": 10, "Cedar": 10, "Vetiver": 10}),
    ("Chanel", "Chance", 2002, "Women", "floral",
        {"Pink Pepper": 10, "Lemon": 10, "Pineapple": 10},
        {"Jasmine": 13, "Hyacinth": 14, "White Musk": 13},
        {"Amber": 10, "Patchouli": 10, "Vanilla": 10}),
    ("Chanel", "Allure Homme Sport", 2004, "Men", "fresh",
        {"Orange": 10, "Mandarin": 10, "Neroli": 10},
        {"Black Pepper": 13, "Cedar": 14, "Tonka Bean": 13},
        {"Amber": 10, "Vetiver": 10, "Musk": 10}),

    # Dior
    ("Dior", "Sauvage", 2015, "Men", "fresh",
        {"Calabrian Bergamot": 15, "Pepper": 15},
        {"Lavender": 13, "Pink Pepper": 13, "Vetiver": 14},
        {"Ambroxan": 10, "Cedar": 10, "Patchouli": 10}),
    ("Dior", "J'adore", 1999, "Women", "floral",
        {"Bergamot": 10, "Mandarin": 10, "Ivy Leaf": 10},
        {"Jasmine": 15, "Rose": 15, "Orchid": 10},
        {"Amaranth": 10, "Blackberry": 10, "Plum": 10}),
    ("Dior", "Miss Dior", 2012, "Women", "floral",
        {"Blood Orange": 10, "Mandarin": 10, "Pink Pepper": 10},
        {"Rose": 15, "Peony": 15, "Lily": 10},
        {"Patchouli": 10, "Musk": 10, "Amber": 10}),
    ("Dior", "Fahrenheit", 1988, "Men", "oriental",
        {"Nutmeg": 10, "Lavender": 10, "Hawthorn": 10},
        {"Violet Leaf": 13, "Nutmeg": 14, "Cedar": 13},
        {"Leather": 10, "Vetiver": 10, "Musk": 10}),
    ("Dior", "Homme Intense", 2011, "Men", "woody",
        {"Lavender": 15, "Bergamot": 15},
        {"Iris": 15, "Pear": 12, "Amber": 13},
        {"Vetiver": 10, "Cedar": 10, "Sandalwood": 10}),

    # Tom Ford
    ("Tom Ford", "Black Orchid", 2006, "Unisex", "oriental",
        {"Truffle": 10, "Bergamot": 10, "Ylang-Ylang": 10},
        {"Black Orchid": 15, "Lotus": 12, "Fruity Notes": 13},
        {"Patchouli": 10, "Vanilla": 10, "Incense": 10}),
    ("Tom Ford", "Oud Wood", 2007, "Unisex", "woody",
        {"Rosewood": 10, "Cardamom": 10, "Chinese Pepper": 10},
        {"Oud": 15, "Sandalwood": 15, "Vetiver": 10},
        {"Tonka Bean": 10, "Amber": 10, "Vanilla": 10}),
    ("Tom Ford", "Tobacco Vanille", 2007, "Unisex", "oriental",
        {"Tobacco Leaf": 15, "Spicy Notes": 15},
        {"Vanilla": 15, "Cacao": 12, "Tonka Bean": 13},
        {"Dried Fruits": 10, "Woody Notes": 10, "Sweet Notes": 10}),
    ("Tom Ford", "Neroli Portofino", 2011, "Unisex", "fresh",
        {"Neroli": 12, "Bergamot": 10, "Lemon": 8},
        {"African Orange Flower": 13, "Jasmine": 14, "Lavender": 13},
        {"Amber": 10, "Angelica": 10, "Musk": 10}),
    ("Tom Ford", "Lost Cherry", 2018, "Unisex", "oriental",
        {"Cherry": 15, "Bitter Almond": 15},
        {"Turkish Rose": 13, "Jasmine Sambac": 14, "Plum": 13},
        {"Tonka Bean": 10, "Sandalwood": 10, "Vetiver": 10}),

    # YSL (Yves Saint Laurent)
    ("YSL", "Black Opium", 2014, "Women", "oriental",
        {"Pink Pepper": 10, "Orange": 10, "Pear": 10},
        {"Coffee": 15, "Jasmine": 12, "Bitter Almond": 13},
        {"Vanilla": 12, "Patchouli": 10, "Cedar": 8}),
    ("YSL", "Y", 2017, "Men", "aromatic",
        {"Apple": 10, "Ginger": 10, "Bergamot": 10},
        {"Sage": 13, "Geranium": 14, "Violet Leaf": 13},
        {"Cedar": 10, "Tonka Bean": 10, "Vetiver": 10}),
    ("YSL", "La Nuit de L'Homme", 2009, "Men", "woody",
        {"Cardamom": 10, "Bergamot": 10, "Lavender": 10},
        {"Cedar": 13, "Vetiver": 14, "Cumin": 13},
        {"Coumarin": 10, "Caraway": 10, "Oud": 10}),
    ("YSL", "Mon Paris", 2016, "Women", "floral",
        {"Strawberry": 10, "Raspberry": 10, "Pear": 10},
        {"Peony": 13, "Datura": 14, "Orange Blossom": 13},
        {"Patchouli": 10, "White Musk": 10, "Amber": 10}),
    ("YSL", "Libre", 2019, "Women", "floral",
        {"Mandarin": 10, "Lavender": 10, "Black Currant": 10},
        {"Orange Blossom": 13, "Jasmine": 14, "Lavender": 13},
        {"Vanilla": 10, "Musk": 10, "Amber": 10}),

    # Giorgio Armani
    ("Giorgio Armani", "Acqua di Gio", 1996, "Men", "fresh",
        {"Lime": 10, "Lemon": 10, "Bergamot": 10},
        {"Jasmine": 13, "Calone": 14, "Rosemary": 13},
        {"Cedar": 10, "Musk": 10, "Patchouli": 10}),
    ("Giorgio Armani", "Si", 2013, "Women", "floral",
        {"Black Currant": 15, "Mandarin": 15},
        {"May Rose": 13, "Freesia": 14, "Jasmine": 13},
        {"Patchouli": 10, "Vanilla": 10, "Amber": 10}),
    ("Giorgio Armani", "Code", 2004, "Men", "oriental",
        {"Lemon": 10, "Bergamot": 10, "Anise": 10},
        {"Olive Blossom": 13, "Star Anise": 14, "Guaiac Wood": 13},
        {"Tonka Bean": 10, "Leather": 10, "Tobacco": 10}),
    ("Giorgio Armani", "My Way", 2020, "Women", "floral",
        {"Orange Blossom": 10, "Bergamot": 10, "Indian Tuberose": 10},
        {"Jasmine": 13, "Tuberose": 14, "Orange Blossom": 13},
        {"Vanilla": 10, "White Musk": 10, "Cedar": 10}),
    ("Giorgio Armani", "Stronger With You", 2017, "Men", "oriental",
        {"Cardamom": 10, "Pink Pepper": 10, "Violet Leaf": 10},
        {"Sage": 13, "Meringue": 14, "Cinnamon": 13},
        {"Vanilla": 10, "Tonka Bean": 10, "Chestnut": 10}),

    # Gucci
    ("Gucci", "Guilty", 2010, "Women", "floral",
        {"Mandarin": 10, "Pink Pepper": 10, "Bergamot": 10},
        {"Geranium": 13, "Peach": 14, "Lilac": 13},
        {"Patchouli": 10, "Amber": 10, "Vanilla": 10}),
    ("Gucci", "Bloom", 2017, "Women", "floral",
        {"Rangoon Creeper": 15, "Tuberose": 15},
        {"Jasmine": 13, "Honeysuckle": 14, "Natural Tuberose": 13},
        {"Orris Root": 10, "Sandalwood": 10, "Benzoin": 10}),
    ("Gucci", "Guilty Absolute", 2017, "Men", "woody",
        {"Leather": 15, "Wood": 15},
        {"Patchouli": 13, "Goldenwood": 14, "Vetiver": 13},
        {"Nootka Tree": 10, "Cypress": 10, "Cedar": 10}),
    ("Gucci", "Flora", 2009, "Women", "floral",
        {"Citrus": 10, "Peony": 10, "Mandarin": 10},
        {"Rose": 13, "Osmanthus": 14, "Pink Pepper": 13},
        {"Sandalwood": 10, "Patchouli": 10, "Pink Pepper": 10}),
    ("Gucci", "Bamboo", 2015, "Women", "floral",
        {"Bergamot": 15, "Orange Blossom": 15},
        {"Casablanca Lily": 13, "Ylang-Ylang": 14, "Orange Blossom": 13},
        {"Tahitian Vanilla": 10, "Sandalwood": 10, "Amber": 10}),

    # Paco Rabanne
    ("Paco Rabanne", "1 Million", 2008, "Men", "oriental",
        {"Grapefruit": 10, "Mint": 10, "Blood Mandarin": 10},
        {"Cinnamon": 13, "Rose": 14, "Spicy Notes": 13},
        {"Leather": 10, "Amber": 10, "Patchouli": 10}),
    ("Paco Rabanne", "Invictus", 2013, "Men", "fresh",
        {"Grapefruit": 10, "Marine Notes": 10, "Mandarin": 10},
        {"Bay Leaf": 13, "Jasmine": 14, "Hedione": 13},
        {"Guaiac Wood": 10, "Patchouli": 10, "Ambergris": 10}),
    ("Paco Rabanne", "Lady Million", 2010, "Women", "floral",
        {"Neroli": 10, "Raspberry": 10, "Amalfi Lemon": 10},
        {"African Orange Flower": 13, "Jasmine": 14, "Gardenia": 13},
        {"Honey": 10, "Patchouli": 10, "Amber": 10}),
    ("Paco Rabanne", "Phantom", 2021, "Men", "aromatic",
        {"Lemon": 10, "Lavender": 10, "Cardamom": 10},
        {"Earthy Notes": 13, "Lavender": 14, "Patchouli": 13},
        {"Vanilla": 10, "Vetiver": 10, "Woody Notes": 10}),
    ("Paco Rabanne", "Olympea", 2015, "Women", "oriental",
        {"Green Mandarin": 10, "Water Jasmine": 10, "Ginger Flower": 10},
        {"Salted Vanilla": 13, "Jasmine": 14, "Ginger": 13},
        {"Cashmere Wood": 10, "Sandalwood": 10, "Ambergris": 10}),

    # Viktor & Rolf
    ("Viktor & Rolf", "Flowerbomb", 2005, "Women", "floral",
        {"Tea": 10, "Bergamot": 10, "Osmanthus": 10},
        {"Sambac Jasmine": 13, "Orchid": 14, "Freesia": 13},
        {"Patchouli": 10, "Musk": 10, "Amber": 10}),
    ("Viktor & Rolf", "Spicebomb", 2012, "Men", "oriental",
        {"Pink Pepper": 10, "Elemi": 10, "Bergamot": 10},
        {"Cinnamon": 13, "Saffron": 14, "Paprika": 13},
        {"Tobacco": 10, "Vetiver": 10, "Leather": 10}),
    ("Viktor & Rolf", "Bonbon", 2014, "Women", "oriental",
        {"Mandarin": 10, "Orange": 10, "Peach": 10},
        {"Caramel": 13, "Orange Blossom": 14, "Jasmine": 13},
        {"Amber": 10, "Cedar": 10, "Sandalwood": 10}),
    ("Viktor & Rolf", "Flowerbomb Dew", 2019, "Women", "floral",
        {"Pomegranate": 15, "Mandarin": 15},
        {"Iris": 13, "Peony": 14, "Rose": 13},
        {"Amberwood": 10, "Musk": 10, "Cashmere Wood": 10}),
    ("Viktor & Rolf", "Spicebomb Extreme", 2015, "Men", "oriental",
        {"Caraway": 10, "Black Pepper": 10, "Grapefruit": 10},
        {"Cinnamon": 13, "Saffron": 14, "Cumin": 13},
        {"Tobacco": 10, "Vanilla": 10, "Bourbon Vetiver": 10}),

    # Dolce & Gabbana
    ("Dolce & Gabbana", "Light Blue", 2001, "Women", "fresh",
        {"Sicilian Lemon": 10, "Apple": 10, "Bluebell": 10},
        {"Jasmine": 13, "Bamboo": 14, "White Rose": 13},
        {"Cedar": 10, "Amber": 10, "Musk": 10}),
    ("Dolce & Gabbana", "The One", 2006, "Women", "oriental",
        {"Bergamot": 10, "Mandarin": 10, "Lychee": 10},
        {"Peach": 13, "Plum": 14, "Jasmine": 13},
        {"Vanilla": 10, "Amber": 10, "Musk": 10}),
    ("Dolce & Gabbana", "K", 2019, "Men", "aromatic",
        {"Blood Orange": 10, "Sicilian Lemon": 10, "Juniper": 10},
        {"Lavender": 13, "Geranium": 14, "Clary Sage": 13},
        {"Vetiver": 10, "Patchouli": 10, "Cedar": 10}),
    ("Dolce & Gabbana", "Dolce", 2014, "Women", "floral",
        {"Papaya Flower": 10, "Neroli": 10, "Water Lily": 10},
        {"Narcissus": 13, "White Amaryllis": 14, "White Water Lily": 13},
        {"Musk": 10, "Cashmere Wood": 10, "Sandalwood": 10}),
    ("Dolce & Gabbana", "Intenso", 2014, "Men", "aromatic",
        {"Basil": 10, "Lavender": 10, "Marigold": 10},
        {"Geranium": 13, "Hay": 14, "Moepel Accord": 13},
        {"Tobacco": 10, "Sandalwood": 10, "Labdanum": 10}),

    # Lancôme
    ("Lancome", "La Vie Est Belle", 2012, "Women", "floral",
        {"Black Currant": 10, "Pear": 10, "Pink Pepper": 10},
        {"Iris": 13, "Jasmine": 14, "Orange Blossom": 13},
        {"Vanilla": 10, "Patchouli": 10, "Tonka Bean": 10}),
    ("Lancome", "Trésor", 1990, "Women", "floral",
        {"Pineapple": 10, "Peach": 10, "Rose": 10},
        {"Iris": 13, "Lily": 14, "Heliotrope": 13},
        {"Musk": 10, "Sandalwood": 10, "Amber": 10}),
    ("Lancome", "Hypnôse", 2005, "Women", "oriental",
        {"Passion Fruit": 15, "Apricot": 15},
        {"Jasmine": 13, "Lily": 14, "Gardenia": 13},
        {"Vanilla": 10, "Vetiver": 10, "Musk": 10}),
    ("Lancome", "Idôle", 2019, "Women", "floral",
        {"Bergamot": 10, "Pear": 10, "Pink Pepper": 10},
        {"Rose": 13, "Jasmine": 14, "Indian Jasmine": 13},
        {"White Musk": 10, "Vanilla": 10, "Patchouli": 10}),
    ("Lancome", "La Nuit Trésor", 2015, "Women", "oriental",
        {"Pear": 10, "Raspberry": 10, "Lychee": 10},
        {"Rose": 13, "Jasmine": 14, "Peach": 13},
        {"Vanilla": 10, "Tonka Bean": 10, "Patchouli": 10}),

    # Burberry
    ("Burberry", "Brit", 2003, "Women", "floral",
        {"Lime": 10, "Pear": 10, "Almond": 10},
        {"Peony": 13, "Sugar": 14, "Candied Almond": 13},
        {"Vanilla": 10, "Tonka Bean": 10, "Amber": 10}),
    ("Burberry", "London", 2006, "Men", "fresh",
        {"Lavender": 10, "Cinnamon": 10, "Bergamot": 10},
        {"Mimosa": 13, "Leather": 14, "Port Wine": 13},
        {"Tobacco Leaf": 10, "Guaiac Wood": 10, "Oakmoss": 10}),
    ("Burberry", "Her", 2018, "Women", "floral",
        {"Red Berries": 10, "Blackberry": 10, "Sour Cherry": 10},
        {"Jasmine": 13, "Violet": 14, "Peony": 13},
        {"Musk": 10, "Amber": 10, "Dry Woody Notes": 10}),
    ("Burberry", "Mr. Burberry", 2016, "Men", "woody",
        {"Grapefruit": 10, "Cardamom": 10, "Tarragon": 10},
        {"Birch Leaf": 13, "Nutmeg": 14, "Cedar": 13},
        {"Vetiver": 10, "Guaiac Wood": 10, "Sandalwood": 10}),
    ("Burberry", "Weekend", 1997, "Women", "floral",
        {"Reseda": 10, "Tangerine": 10, "Wild Rose": 10},
        {"Peach": 13, "Hyacinth": 14, "Iris": 13},
        {"Sandalwood": 10, "Musk": 10, "Amber": 10}),

    # Hermès
    ("Hermes", "Terre d'Hermès", 2006, "Men", "woody",
        {"Orange": 10, "Grapefruit": 10, "Pepper": 10},
        {"Pelargonium": 13, "Flint": 14, "Baies Rose": 13},
        {"Vetiver": 10, "Cedar": 10, "Benzoin": 10}),
    ("Hermes", "Twilly d'Hermès", 2017, "Women", "floral",
        {"Ginger": 15, "Bergamot": 15},
        {"Tuberose": 13, "Orange Blossom": 14, "Jasmine": 13},
        {"Sandalwood": 10, "Vanilla": 10, "Amber": 10}),
    ("Hermes", "Un Jardin sur le Nil", 2005, "Unisex", "fresh",
        {"Grapefruit": 10, "Green Mango": 10, "Tomato Leaf": 10},
        {"Lotus": 13, "Calamus": 14, "Hyacinth": 13},
        {"Sycamore Wood": 10, "Musk": 10, "Incense": 10}),
    ("Hermes", "Voyage d'Hermès", 2010, "Unisex", "woody",
        {"Cardamom": 10, "Lemon": 10, "Tea": 10},
        {"Green Notes": 13, "Hedione": 14, "Rose": 13},
        {"Cedar": 10, "Amber": 10, "Musk": 10}),
    ("Hermes", "Eau des Merveilles", 2004, "Women", "oriental",
        {"Orange": 10, "Elemi": 10, "Lemon": 10},
        {"Amber": 13, "Pepper": 14, "Pink Pepper": 13},
        {"Fir Resin": 10, "Cedar": 10, "Oakmoss": 10}),

    # Versace
    ("Versace", "Eros", 2012, "Men", "oriental",
        {"Mint": 10, "Green Apple": 10, "Lemon": 10},
        {"Tonka Bean": 13, "Geranium": 14, "Ambroxan": 13},
        {"Vanilla": 10, "Vetiver": 10, "Oakmoss": 10}),
    ("Versace", "Bright Crystal", 2006, "Women", "floral",
        {"Pomegranate": 10, "Yuzu": 10, "Iced Accord": 10},
        {"Peony": 13, "Lotus Flower": 14, "Magnolia": 13},
        {"Amber": 10, "Musk": 10, "Mahogany": 10}),
    ("Versace", "Dylan Blue", 2016, "Men", "aromatic",
        ("Calabrian Bergamot": 10, "Grapefruit": 10, "Fig Leaves": 10},
        {"Violet Leaf": 13, "Papyrus": 14, "Black Pepper": 13},
        {"Mineral Musk": 10, "Tonka Bean": 10, "Incense": 10}),
    ("Versace", "Crystal Noir", 2004, "Women", "oriental",
        {"Ginger": 10, "Cardamom": 10, "Pepper": 10},
        {"Gardenia": 13, "Orange Blossom": 14, "Peony": 13},
        {"Sandalwood": 10, "Musk": 10, "Amber": 10}),
    ("Versace", "Pour Homme", 2008, "Men", "aromatic",
        {"Lemon": 10, "Neroli": 10, "Bergamot": 10},
        {"Hyacinth": 13, "Cedar": 14, "Clary Sage": 13},
        {"Tonka Bean": 10, "Musk": 10, "Amber": 10}),

    # Calvin Klein
    ("Calvin Klein", "CK One", 1994, "Unisex", "fresh",
        {"Lemon": 10, "Bergamot": 10, "Pineapple": 10},
        {"Jasmine": 13, "Violet": 14, "Rose": 13},
        {"Musk": 10, "Cedar": 10, "Amber": 10}),
    ("Calvin Klein", "Euphoria", 2005, "Women", "oriental",
        {"Pomegranate": 10, "Persimmon": 10, "Green Notes": 10},
        {"Lotus Blossom": 13, "Orchid": 14, "Champaca": 13},
        {"Amber": 10, "Mahogany": 10, "Violet": 10}),
    ("Calvin Klein", "Obsession", 1985, "Women", "oriental",
        {"Mandarin": 10, "Vanilla": 10, "Bergamot": 10},
        {"Jasmine": 13, "Orange Blossom": 14, "Spicy Notes": 13},
        {"Amber": 10, "Incense": 10, "Oakmoss": 10}),
    ("Calvin Klein", "Eternity", 1988, "Women", "floral",
        {"Mandarin": 10, "Freesia": 10, "Sage": 10},
        {"Lily": 13, "Marigold": 14, "Narcissus": 13},
        {"Patchouli": 10, "Sandalwood": 10, "Amber": 10}),
    ("Calvin Klein", "CK Be", 1996, "Unisex", "aromatic",
        {"Bergamot": 10, "Mint": 10, "Lavender": 10},
        {"Magnolia": 13, "Peach": 14, "Juniper": 13},
        {"Musk": 10, "Sandalwood": 10, "Opoponax": 10}),

    # Bvlgari
    ("Bvlgari", "Omnia", 2003, "Women", "oriental",
        {"Almond": 10, "Saffron": 10, "Ginger": 10},
        {"Lotus": 13, "Nutmeg": 14, "Cinnamon": 13},
        {"Sandalwood": 10, "Chocolate": 10, "Vanilla": 10}),
    ("Bvlgari", "Aqva", 2005, "Men", "fresh",
        {"Mandarin": 10, "Neroli": 10, "Petitgrain": 10},
        {"Posidonia": 13, "Mineral Amber": 14, "Santolina": 13},
        {"Amber": 10, "Cedar": 10, "Clary Sage": 10}),
    ("Bvlgari", "Jasmin Noir", 2008, "Women", "floral",
        {"Gardenia": 15, "Green Sap": 15},
        {"Sambac Jasmine": 13, "Lily": 14, "Almond": 13},
        {"Tonka Bean": 10, "Precious Woods": 10, "Musk": 10}),
    ("Bvlgari", "Man in Black", 2014, "Men", "oriental",
        {"Rum": 10, "Natural Spices": 10, "Tobacco": 10},
        {"Tuberose": 13, "Iris": 14, "Leather": 13},
        {"Benzoin": 10, "Tonka Bean": 10, "Guaiac Wood": 10}),
    ("Bvlgari", "Rose Goldea", 2016, "Women", "floral",
        {"Pomegranate": 10, "Damask Rose": 10, "Jasmine": 10},
        {"Rosa Damascena": 13, "Musk": 14, "Incense": 13},
        {"Sandalwood": 10, "Musk": 10, "Olibanum": 10}),

    # Givenchy
    ("Givenchy", "L'Interdit", 2018, "Women", "floral",
        {"Bergamot": 10, "Pear": 10, "Mandarin": 10},
        {"Tuberose": 13, "Orange Blossom": 14, "Jasmine Sambac": 13},
        {"Patchouli": 10, "Vanilla": 10, "Ambroxan": 10}),
    ("Givenchy", "Gentleman", 2017, "Men", "aromatic",
        {"Pear": 10, "Cardamom": 10, "Lavender": 10},
        {"Iris": 13, "Geranium": 14, "Cinnamon": 13},
        {"Leather": 10, "Patchouli": 10, "Black Vanilla": 10}),
    ("Givenchy", "Ange ou Démon", 2006, "Women", "oriental",
        {"Mandarin": 10, "Saffron": 10, "Thyme": 10},
        {"Lily": 13, "Orchid": 14, "Ylang-Ylang": 13},
        {"Tonka Bean": 10, "Rosewood": 10, "Vanilla": 10}),
    ("Givenchy", "Play", 2008, "Men", "aromatic",
        {"Mandarin": 10, "Bergamot": 10, "Grapefruit": 10},
        {"Black Pepper": 13, "Coffee": 14, "Amyris": 13},
        {"Patchouli": 10, "Vetiver": 10, "Cedar": 10}),
    ("Givenchy", "Live Irrésistible", 2015, "Women", "floral",
        {"Pineapple": 10, "Blackberry": 10, "Rose": 10},
        {"Blond Woods": 13, "Orange Blossom": 14, "Angelica Root": 13},
        {"Patchouli": 10, "Cashmere Wood": 10, "Musk": 10}),

    # Prada
    ("Prada", "Candy", 2011, "Women", "oriental",
        {"Caramel": 15, "Musk": 15},
        {"Powder": 13, "Benzoin": 14, "Vanilla": 13},
        {"Caramel": 10, "Benzoin": 10, "Musk": 10}),
    ("Prada", "L'Homme", 2016, "Men", "aromatic",
        {"Neroli": 10, "Black Pepper": 10, "Carrot Seeds": 10},
        {"Iris": 13, "Geranium": 14, "Violet": 13},
        {"Patchouli": 10, "Cedar": 10, "Amber": 10}),
    ("Prada", "La Femme", 2016, "Women", "floral",
        {"Frangipani": 15, "Ylang-Ylang": 15},
        {"Beeswax": 13, "Tuberose": 14, "Vanilla": 13},
        {"Benzoin": 10, "Tonka Bean": 10, "Vetiver": 10}),
    ("Prada", "Luna Rossa", 2012, "Men", "aromatic",
        {"Lavender": 10, "Bitter Orange": 10, "Mint": 10},
        {"Clary Sage": 13, "Spearmint": 14, "Orange Blossom": 13},
        {"Amber": 10, "Musk": 10, "Ambrette": 10}),
    ("Prada", "Infusion d'Iris", 2007, "Women", "floral",
        {"Neroli": 10, "Mandarin": 10, "Orange Blossom": 10},
        {"Iris": 13, "Galbanum": 14, "Vetiver": 13},
        {"Benzoin": 10, "Incense": 10, "Cedar": 10}),

    # Marc Jacobs
    ("Marc Jacobs", "Daisy", 2007, "Women", "floral",
        {"Blood Grapefruit": 10, "Strawberry": 10, "Violet Leaves": 10},
        {"Gardenia": 13, "Violet": 14, "Jasmine": 13},
        {"Musk": 10, "Vanilla": 10, "White Woods": 10}),
    ("Marc Jacobs", "Decadence", 2015, "Women", "oriental",
        {"Plum": 10, "Saffron": 10, "Iris": 10},
        {"Bulgarian Rose": 13, "Jasmine Sambac": 14, "Orris Root": 13},
        {"Papyrus": 10, "Vetiver": 10, "Amber": 10}),
    ("Marc Jacobs", "Dot", 2012, "Women", "floral",
        {"Red Berries": 10, "Dragonfruit": 10, "Honeysuckle": 10},
        {"Jasmine": 13, "Coconut Water": 14, "Orange Blossom": 13},
        {"Musk": 10, "Vanilla": 10, "Driftwood": 10}),
    ("Marc Jacobs", "Bang", 2010, "Men", "woody",
        {"Pink Pepper": 10, "Fennel": 10, "Pepper": 10},
        {"Davana": 13, "Patchouli": 14, "Vetiver": 13},
        {"White Woods": 10, "Moss": 10, "Tonka Bean": 10}),
    ("Marc Jacobs", "Perfect", 2020, "Women", "floral",
        {"Rhubarb": 10, "Narcissus": 10, "Almond Milk": 10},
        {"Daffodil": 13, "Narcissus": 14, "Almond Milk": 13},
        {"Cedar": 10, "Cashmeran": 10, "Musk": 10}),

    # Thierry Mugler
    ("Thierry Mugler", "Angel", 1992, "Women", "oriental",
        {"Melon": 10, "Coconut": 10, "Mandarin": 10},
        {"Honey": 13, "Jasmine": 14, "Red Berries": 13},
        {"Patchouli": 10, "Chocolate": 10, "Vanilla": 10}),
    ("Thierry Mugler", "Alien", 2005, "Women", "oriental",
        {"Jasmine Sambac": 15, "Cashmeran Wood": 15},
        {"Jasmine": 13, "Amber": 14, "White Amber": 13},
        {"White Amber": 10, "Woodsy Notes": 10, "Vanilla": 10}),
    ("Thierry Mugler", "A*Men", 1996, "Men", "oriental",
        {"Coffee": 10, "Lavender": 10, "Bergamot": 10},
        {"Patchouli": 13, "Honey": 14, "Helional": 13},
        {"Vanilla": 10, "Tonka Bean": 10, "Amber": 10}),
    ("Thierry Mugler", "Aura", 2017, "Women", "floral",
        {"Rhubarb Leaf": 10, "Orange Blossom": 10, "Lemon": 10},
        {"Orange Blossom": 13, "Neroli": 14, "Tiger Orchid": 13},
        {"Bourbon Vanilla": 10, "Wolfwood": 10, "Amber": 10}),
    ("Thierry Mugler", "Cologne", 2001, "Unisex", "fresh",
        {"African Orange Flower": 10, "Bergamot": 10, "Petitgrain": 10},
        {"Neroli": 13, "White Musks": 14, "Green Notes": 13},
        {"Musk": 10, "White Musk": 10, "Neroli": 10}),

    # Montblanc
    ("Montblanc", "Legend", 2011, "Men", "aromatic",
        {"Bergamot": 10, "Lavender": 10, "Pineapple": 10},
        {"Coumarin": 13, "Evernyl": 14, "Geranium": 13},
        {"Sandalwood": 10, "Tonka Bean": 10, "Oakmoss": 10}),
    ("Montblanc", "Explorer", 2019, "Men", "woody",
        {"Bergamot": 10, "Pink Pepper": 10, "Clary Sage": 10},
        {"Haitian Vetiver": 13, "Leather": 14, "Akigalawood": 13},
        {"Ambroxan": 10, "Indonesian Patchouli": 10, "Cacao Pod": 10}),
    ("Montblanc", "Emblem", 2014, "Men", "woody",
        {"Clary Sage": 10, "Cardamom": 10, "Grapefruit": 10},
        {"Cinnamon": 13, "Violet Leaves": 14, "Frozen Violet": 13},
        {"Tonka Bean": 10, "Woody Notes": 10, "Precious Woods": 10}),
    ("Montblanc", "Lady Emblem", 2015, "Women", "floral",
        {"Pink Pepper": 10, "Pink Grapefruit": 10, "Star Fruit": 10},
        {"Rose": 13, "Jasmine": 14, "Orange Blossom": 13},
        {"Musk": 10, "Sandalwood": 10, "Patchouli": 10}),
    ("Montblanc", "Legend Spirit", 2016, "Men", "aromatic",
        {"Grapefruit": 10, "Pink Pepper": 10, "Bergamot": 10},
        {"Lavender": 13, "Cardamom": 14, "Aquatic Accord": 13},
        {"White Woods": 10, "Oakmoss": 10, "Cashmere Wood": 10}),

    # Carolina Herrera
    ("Carolina Herrera", "Good Girl", 2016, "Women", "oriental",
        {"Almond": 10, "Coffee": 10, "Bergamot": 10},
        {"Tuberose": 13, "Jasmine Sambac": 14, "Bulgarian Rose": 13},
        {"Tonka Bean": 10, "Cacao": 10, "Sandalwood": 10}),
    ("Carolina Herrera", "212 VIP", 2010, "Women", "floral",
        {"Passion Fruit": 10, "Rum": 10, "Vodka": 10},
        {"Gardenia": 13, "Musk": 14, "Vanilla Orchid": 13},
        {"Amber": 10, "Musk": 10, "Tonka Bean": 10}),
    ("Carolina Herrera", "Bad Boy", 2019, "Men", "oriental",
        {"Black Pepper": 10, "White Pepper": 10, "Bergamot": 10},
        {"Sage": 13, "Cedar": 14, "Vetiver": 13},
        {"Tonka Bean": 10, "Cocoa": 10, "Amberwood": 10}),
    ("Carolina Herrera", "CH Men", 2009, "Men", "aromatic",
        {"Grapefruit": 10, "Bergamot": 10, "Lemon": 10},
        {"Violet Leaf": 13, "Nutmeg": 14, "Jasmine": 13},
        {"Sandalwood": 10, "Leather": 10, "Amber": 10}),
    ("Carolina Herrera", "212 Men", 1999, "Men", "woody",
        {"Green Notes": 10, "Petitgrain": 10, "Spices": 10},
        {"Ginger": 13, "Violet Leaf": 14, "Gardenia": 13},
        {"Sandalwood": 10, "Incense": 10, "Labdanum": 10}),

    # Miu Miu
    ("Miu Miu", "Miu Miu", 2015, "Women", "floral",
        {"Lily of the Valley": 15, "Green Notes": 15},
        {"Jasmine Absolute": 13, "Cashmere Wood": 14, "Rose Absolute": 13},
        {"Akigalawood": 10, "Pink Pepper": 10, "Patchouli": 10}),
    ("Miu Miu", "L'Eau Rosée", 2017, "Women", "floral",
        {"Cassis": 10, "Lily of the Valley": 10, "Ginger": 10},
        {"Rose": 13, "Pink Peony": 14, "Hawthorn": 13},
        {"Cedarwood": 10, "Cashmere Wood": 10, "White Musk": 10}),
    ("Miu Miu", "L'Eau Bleue", 2018, "Women", "floral",
        {"Cassis": 10, "Lily of the Valley": 10, "Apricot": 10},
        {"Peony": 13, "Rose": 14, "Hawthorn": 13},
        {"Akigalawood": 10, "Cashmere Wood": 10, "White Musk": 10}),
    ("Miu Miu", "Twist", 2021, "Women", "floral",
        {"Apple Blossom": 10, "Bergamot": 10, "Cedar Leaves": 10},
        {"Pink Amber": 13, "Jasmine": 14, "Cedar": 13},
        {"Benzoin": 10, "Tonka Bean": 10, "Pink Amber": 10}),

    # Chloe
    ("Chloe", "Chloe", 2008, "Women", "floral",
        {"Pink Peony": 10, "Freesia": 10, "Lychee": 10},
        {"Magnolia": 13, "Lily of the Valley": 14, "Rose": 13},
        {"Amber": 10, "Cedar": 10, "Honey": 10}),
    ("Chloe", "Love Story", 2014, "Women", "floral",
        {"Neroli": 10, "Orange Blossom": 10, "Stephanotis": 10},
        {"Stephanotis": 13, "Orange Blossom": 14, "Jasmine Sambac": 13},
        {"Cedarwood": 10, "Musk": 10, "Benzoin": 10}),
    ("Chloe", "Nomade", 2018, "Women", "floral",
        {"Mirabelle": 10, "Lemon": 10, "Bergamot": 10},
        {"Freesia": 13, "Peach": 14, "Plum": 13},
        {"Oakmoss": 10, "Patchouli": 10, "White Musk": 10}),
    ("Chloe", "Roses de Chloe", 2013, "Women", "floral",
        {"Bergamot": 10, "Lychee": 10, "Damask Rose": 10},
        {"Magnolia": 13, "Damask Rose": 14, "White Rose": 13},
        {"Amber": 10, "White Musk": 10, "Virginian Cedar": 10}),
    ("Chloe", "L'Eau", 2019, "Women", "floral",
        {"Rose": 10, "Grapefruit": 10, "Lychee": 10},
        {"Damask Rose": 13, "Verbena": 14, "Magnolia": 13},
        {"Cedar": 10, "Amber": 10, "Musk": 10}),

    # Azzaro
    ("Azzaro", "Wanted", 2016, "Men", "fresh",
        {"Lemon": 10, "Ginger": 10, "Lavender": 10},
        {"Apple": 13, "Juniper": 14, "Cardamom": 13},
        {"Tonka Bean": 10, "Amberwood": 10, "Haitian Vetiver": 10}),
    ("Azzaro", "Chrome", 1996, "Men", "fresh",
        {"Lemon": 10, "Rosemary": 10, "Bergamot": 10},
        {"Cyclamen": 13, "Coriander": 14, "Oakmoss": 13},
        {"Musk": 10, "Cedar": 10, "Sandalwood": 10}),
    ("Azzaro", "The Most Wanted", 2021, "Men", "oriental",
        {"Cardamom": 10, "Red Ginger": 10, "Mint": 10},
        {"Bourbon Vanilla": 13, "Toffee Accord": 14, "Amberwood": 13},
        {"Tonka Bean": 10, "Woody Notes": 10, "Warm Spices": 10}),
    ("Azzaro", "Azzaro Pour Homme", 1978, "Men", "aromatic",
        {"Lavender": 10, "Star Anise": 10, "Caraway": 10},
        {"Basil": 13, "Geranium": 14, "Iris": 13},
        {"Vetiver": 10, "Oakmoss": 10, "Cedar": 10}),
    ("Azzaro", "Wanted Girl", 2019, "Women", "oriental",
        {"Orange Blossom": 10, "Ginger Flower": 10, "Pink Pepper": 10},
        {"Pomegranate": 13, "Dulce de Leche": 14, "Datura": 13},
        {"Tonka Bean": 10, "Haitian Vetiver": 10, "Patchouli": 10}),

    # Bottega Veneta
    ("Bottega Veneta", "Bottega Veneta", 2011, "Women", "floral",
        {"Calabrian Bergamot": 15, "Pink Peppercorn": 15},
        {"Jasmine Sambac": 13, "Brazilian Gardenia": 14, "Boronia": 13},
        {"Patchouli": 10, "Leather Accord": 10, "Oakmoss": 10}),
    ("Bottega Veneta", "Illusione", 2015, "Women", "floral",
        {"Bitter Orange": 10, "Pink Pepper": 10, "Neroli": 10},
        {"Orange Blossom": 13, "Jasmine Sambac": 14, "Tuberose": 13},
        {"Ambrette": 10, "Musk": 10, "Sandalwood": 10}),
    ("Bottega Veneta", "Pour Homme", 2013, "Men", "woody",
        ("Bergamot": 10, "Pine": 10, "Juniper": 10},
        {"Fir Balsam": 13, "Labdanum": 14, "Pimento": 13},
        {"Leather": 10, "Patchouli": 10, "Papyrus": 10}),

    # Loewe
    ("Loewe", "001 Woman", 2016, "Women", "floral",
        {"Tangerine": 10, "Musk": 10, "Iris": 10},
        {"Jasmine": 13, "Peony": 14, "Lily of the Valley": 13},
        {"Patchouli": 10, "Amber": 10, "Musk": 10}),
    ("Loewe", "Solo", 2003, "Men", "aromatic",
        {"Bergamot": 10, "Basil": 10, "Mandarin": 10},
        {"Sandalwood": 13, "Cinnamon": 14, "Cardamom": 13},
        {"Vetiver": 10, "Musk": 10, "Woody Notes": 10}),
    ("Loewe", "Aura", 2019, "Women", "floral",
        {"Mandarin": 10, "Rose": 10, "Iris": 10},
        {"Indian Tuberose": 13, "Rose": 14, "Ylang-Ylang": 13},
        {"Musk": 10, "Benzoin": 10, "Vanilla": 10}),

    # Salvatore Ferragamo
    ("Salvatore Ferragamo", "Signorina", 2011, "Women", "floral",
        {"Pink Pepper": 10, "Currant": 10, "Jasmine": 10},
        {"Tiare Flower": 13, "Heliotrope": 14, "Rose": 13},
        {"Panacotta": 10, "Musk": 10, "Leather": 10}),
    ("Salvatore Ferragamo", "Uomo", 2015, "Men", "aromatic",
        {"Bergamot": 10, "Black Pepper": 10, "Basil": 10},
        {"Tiramisu": 13, "Ambrette": 14, "Geranium": 13},
        {"Leather": 10, "Tonka Bean": 10, "Cashmere Wood": 10}),
    ("Salvatore Ferragamo", "Amo", 2019, "Women", "floral",
        {"Blackberry": 10, "Plum": 10, "Grapefruit": 10},
        {"Jasmine": 13, "Iris Flower": 14, "Rose": 13},
        {"Vanilla": 10, "Musk": 10, "Leather": 10}),

    # Kenzo
    ("Kenzo", "Flower by Kenzo", 2000, "Women", "floral",
        {"Mandarin": 10, "Hawthorn": 10, "Cassis": 10},
        {"Poppy": 13, "Rose": 14, "Wild Rose": 13},
        {"Vanilla": 10, "Musk": 10, "Opoponax": 10}),
    ("Kenzo", "L'Eau par Kenzo", 1996, "Unisex", "fresh",
        {"Mint": 10, "Lotus": 10, "Yuzu": 10},
        {"Lotus": 13, "Peony": 14, "Water Notes": 13},
        {"White Musk": 10, "Vanilla": 10, "Amber": 10}),
    ("Kenzo", "Homme", 1991, "Men", "woody",
        {"Lemon": 10, "Artemisia": 10, "Basil": 10},
        {"Nutmeg": 13, "Pine": 14, "Peony": 13},
        {"Sandalwood": 10, "Cedar": 10, "Tonka Bean": 10}),
    ("Kenzo", "World", 2014, "Women", "floral",
        {"Peony": 10, "Ambrette": 10, "Blood Orange": 10},
        {"Egyptian Jasmine": 13, "Tunisian Orange Blossom": 14, "Bourbon Vanilla": 13},
        {"Vetiver": 10, "Musk": 10, "Ambrette": 10}),

    # Diesel
    ("Diesel", "Only The Brave", 2009, "Men", "oriental",
        {"Lemon": 10, "Mandarin": 10, "Amalfi Lemon": 10},
        {"Cedar": 13, "Coriander": 14, "Violet": 13},
        {"Styrax": 10, "Leather": 10, "Benzoin": 10}),
    ("Diesel", "Fuel for Life", 2007, "Men", "fresh",
        {"Star Anise": 10, "Pink Pepper": 10, "Grapefruit": 10},
        {"Lavender": 13, "Heliotrope": 14, "Raspberry": 13},
        {"Vetiver": 10, "Patchouli": 10, "Tonka Bean": 10}),
    ("Diesel", "Loverdose", 2011, "Women", "oriental",
        {"Star Anise": 10, "Mandarin": 10, "Licorice": 10},
        {"Gardenia": 13, "Jasmine Sambac": 14, "Orange Blossom": 13},
        {"Vanilla": 10, "Woody Notes": 10, "Amber": 10}),

    # Hugo Boss
    ("Hugo Boss", "Boss Bottled", 1998, "Men", "woody",
        {"Apple": 10, "Plum": 10, "Lemon": 10},
        {"Geranium": 13, "Cinnamon": 14, "Mahonial": 13},
        {"Vanilla": 10, "Sandalwood": 10, "Cedar": 10}),
    ("Hugo Boss", "The Scent", 2015, "Men", "oriental",
        {"Ginger": 10, "Lavender": 10, "Mandarin": 10},
        {"Maninka": 13, "Leather": 14, "Exotic Notes": 13},
        {"Woody Notes": 10, "Cocoa": 10, "Vanilla": 10}),
    ("Hugo Boss", "Ma Vie", 2014, "Women", "floral",
        {"Pink Freesia": 10, "Cactus Flower": 10, "Green Notes": 10},
        {"Jasmine": 13, "Rose": 14, "Almond Blossom": 13},
        {"Cedarwood": 10, "Musk": 10, "Benzoin": 10}),
    ("Hugo Boss", "Jour", 2013, "Women", "floral",
        {"Grapefruit": 10, "Lemon": 10, "White Flowers": 10},
        {"Lily of the Valley": 13, "Freesia": 14, "Jasmine": 13},
        {"White Musk": 10, "Woody Notes": 10, "Sandalwood": 10}),

    # Lacoste
    ("Lacoste", "L.12.12 Blanc", 2011, "Men", "fresh",
        {"Grapefruit": 10, "Cardamom": 10, "Rosemary": 10},
        {"Ylang-Ylang": 13, "Tuberose": 14, "Violet Leaves": 13},
        {"Suede": 10, "Cedar": 10, "Vetiver": 10}),
    ("Lacoste", "Pour Femme", 2003, "Women", "floral",
        {"Apple": 10, "Freesia": 10, "Tangerine": 10},
        {"Heliotrope": 13, "Rose": 14, "Violet": 13},
        {"Amber": 10, "Musk": 10, "Sandalwood": 10}),
    ("Lacoste", "L'Homme", 2017, "Men", "woody",
        {"Mandarin": 10, "Quince": 10, "Rhubarb": 10},
        {"Jasmine": 13, "Black Pepper": 14, "Ginger": 13},
        {"Amber": 10, "Woody Notes": 10, "Vanilla": 10}),

    # Issey Miyake
    ("Issey Miyake", "L'Eau d'Issey", 1992, "Women", "fresh",
        {"Lotus": 10, "Rose": 10, "Freesia": 10},
        {"Lily of the Valley": 13, "Peony": 14, "Carnation": 13},
        {"Cedar": 10, "Amber": 10, "Musk": 10}),
    ("Issey Miyake", "Nuit d'Issey", 2014, "Men", "woody",
        {"Bergamot": 10, "Grapefruit": 10, "Black Pepper": 10},
        {"Leather": 13, "Vetiver": 14, "Spices": 13},
        {"Patchouli": 10, "Incense": 10, "Tonka Bean": 10}),
    ("Issey Miyake", "A Scent", 2009, "Unisex", "fresh",
        {"Galbanum": 10, "Verbena": 10, "Hyacinth": 10},
        {"Jasmine": 13, "White Flowers": 14, "Freesia": 13},
        {"Woody Notes": 10, "Musk": 10, "Frankincense": 10}),

    # Elizabeth Arden
    ("Elizabeth Arden", "Green Tea", 1999, "Women", "fresh",
        {"Lemon": 10, "Orange": 10, "Bergamot": 10},
        {"Green Tea": 13, "Jasmine": 14, "Oakmoss": 13},
        {"Musk": 10, "Amber": 10, "Celery Seeds": 10}),
    ("Elizabeth Arden", "Red Door", 1989, "Women", "floral",
        {"Red Rose": 10, "Ylang-Ylang": 10, "Wild Violets": 10},
        {"Jasmine": 13, "Orchid": 14, "Freesia": 13},
        {"Honey": 10, "Sandalwood": 10, "Vetiver": 10}),
    ("Elizabeth Arden", "5th Avenue", 1996, "Women", "floral",
        {"Lilac": 10, "Linden Blossom": 10, "Magnolia": 10},
        {"Jasmine": 13, "Tuberose": 14, "Ylang-Ylang": 13},
        {"Musk": 10, "Iris": 10, "Sandalwood": 10}),

    # Narciso Rodriguez
    ("Narciso Rodriguez", "For Her", 2003, "Women", "floral",
        {"Orange Blossom": 10, "Osmanthus": 10, "African Orange Flower": 10},
        {"Rose": 13, "Peach": 14, "Musk": 13},
        {"Musk": 10, "Amber": 10, "Vanilla": 10}),
    ("Narciso Rodriguez", "For Him", 2007, "Men", "aromatic",
        {"Violet Leaf": 10, "Cardamom": 10, "Basil": 10},
        {"Musk": 13, "Patchouli": 14, "Amber": 13},
        {"Vetiver": 10, "Tonka Bean": 10, "Woody Notes": 10}),
    ("Narciso Rodriguez", "Fleur Musc", 2014, "Women", "floral",
        {"Pink Pepper": 10, "Bergamot": 10, "Mandarin": 10},
        {"Peony": 13, "Rose": 14, "Pink Musk": 13},
        {"Amber": 10, "Woody Notes": 10, "Musk": 10}),

    # Moschino
    ("Moschino", "Toy 2", 2018, "Women", "floral",
        {"Mandarin": 10, "Magnolia": 10, "Apple Blossom": 10},
        {"Peony": 13, "Jasmine": 14, "White Currant": 13},
        {"Sandalwood": 10, "Amber": 10, "Woody Notes": 10}),
    ("Moschino", "Gold Fresh Couture", 2017, "Women", "floral",
        {"Mandarin": 10, "Pear": 10, "Bergamot": 10},
        {"Peony": 13, "Osmanthus": 14, "Passion Flower": 13},
        {"Ambrofix": 10, "Cedarwood": 10, "Labdanum": 10}),

    # Jimmy Choo
    ("Jimmy Choo", "Jimmy Choo", 2011, "Women", "floral",
        {"Sweet Pea": 10, "Mandarin": 10, "Tiger Orchid": 10},
        {"Indonesian Patchouli": 13, "Orchid": 14, "Toffee": 13},
        {"Caramel": 10, "Vibrant Woods": 10, "Patchouli": 10}),
    ("Jimmy Choo", "Man", 2014, "Men", "aromatic",
        {"Lavender": 10, "Mandarin": 10, "Honeydew Melon": 10},
        {"Pink Pepper": 13, "Geranium": 14, "Pineapple Leaf": 13},
        {"Patchouli": 10, "Suede": 10, "Ambery Woods": 10}),
    ("Jimmy Choo", "L'Eau", 2017, "Women", "floral",
        {"Bergamot": 10, "Red Berries": 10, "Nectarine": 10},
        {"Hibiscus": 13, "Peony": 14, "Jasmine": 13},
        {"Cedarwood": 10, "Musk": 10, "White Woods": 10}),

    # Roberto Cavalli
    ("Roberto Cavalli", "Just Cavalli", 2012, "Women", "fresh",
        {"Ginger": 10, "Water Fruits": 10, "Sorbet": 10},
        {"Lily": 13, "Violet": 14, "Amaryllis": 13},
        {"Musk": 10, "Amber": 10, "Tobacco Leaves": 10}),
    ("Roberto Cavalli", "Paradiso", 2015, "Women", "floral",
        {"Bergamot": 10, "Mandarin": 10, "Orange": 10},
        {"Jasmine": 13, "Passionflower": 14, "Almond": 13},
        {"Cypress": 10, "Pine": 10, "Sandalwood": 10}),

    # Escada
    ("Escada", "Magnetism", 2003, "Women", "floral",
        {"Melon": 10, "Litchi": 10, "Red Berries": 10},
        {"Magnolia": 13, "Jasmine": 14, "Caraway": 13},
        {"Musk": 10, "Sandalwood": 10, "Patchouli": 10}),
    ("Escada", "Sentiment", 1997, "Women", "floral",
        {"Cassis": 10, "Orange Blossom": 10, "Rose": 10},
        {"Magnolia": 13, "Honeysuckle": 14, "Jasmine": 13},
        {"Musk": 10, "Sandalwood": 10, "Vanilla": 10}),

    # Davidoff
    ("Davidoff", "Cool Water", 1988, "Men", "fresh",
        {"Lavender": 10, "Mint": 10, "Coriander": 10},
        {"Jasmine": 13, "Geranium": 14, "Neroli": 13},
        {"Musk": 10, "Cedar": 10, "Tobacco": 10}),
    ("Davidoff", "Echo Woman", 2004, "Women", "floral",
        ("Osmanthus": 10, "Grapes": 10, "Pomegranate": 10},
        {"Jasmine": 13, "Water Lily": 14, "Violet": 13},
        {"Iris": 10, "Musk": 10, "Cedar": 10}),

    # Jean Paul Gaultier
    ("Jean Paul Gaultier", "Classique", 1993, "Women", "oriental",
        {"Anis": 10, "Mandarin": 10, "Pear": 10},
        {"Orange Blossom": 13, "Ginger": 14, "Rose": 13},
        {"Amber": 10, "Vanilla": 10, "Musk": 10}),
    ("Jean Paul Gaultier", "Le Male", 1995, "Men", "oriental",
        {"Lavender": 10, "Mint": 10, "Cardamom": 10},
        {"Cinnamon": 13, "Orange Blossom": 14, "Cumin": 13},
        {"Vanilla": 10, "Tonka Bean": 10, "Sandalwood": 10}),
    ("Jean Paul Gaultier", "Scandal", 2017, "Women", "floral",
        {"Blood Orange": 15, "Mandarin": 15},
        {"Honey": 13, "Gardenia": 14, "Orange Blossom": 13},
        {"Caramel": 10, "Patchouli": 10, "Licorice": 10}),
    ("Jean Paul Gaultier", "Kokorico", 2011, "Men", "oriental",
        {"Fig": 10, "Grapefruit": 10, "Orange": 10},
        {"Cacao": 13, "Fenugreek": 14, "Praline": 13},
        {"Patchouli": 10, "Vetiver": 10, "Cedarwood": 10}),

    # Abercrombie & Fitch
    ("Abercrombie & Fitch", "Fierce", 2002, "Men", "fresh",
        {"Lemon": 10, "Orange": 10, "Cardamom": 10},
        {"Petitgrain": 13, "Fir": 14, "Jasmine": 13},
        {"Vetiver": 10, "Musk": 10, "Oakmoss": 10}),
    ("Abercrombie & Fitch", "8", 2002, "Women", "floral",
        {"Mandarin": 10, "Peach": 10, "Raspberry": 10},
        {"Lily of the Valley": 13, "Ylang-Ylang": 14, "Red Poppy": 13},
        {"Vanilla": 10, "Musk": 10, "Sandalwood": 10}),

    # Hollister
    ("Hollister", "Wave", 2010, "Men", "fresh",
        {"Sage": 10, "Mimosa": 10, "Coconut": 10},
        {"Neroli": 13, "Salted Accord": 14, "Violet": 13},
        {"Driftwood": 10, "Musk": 10, "Amber": 10}),
    ("Hollister", "Festival Vibes", 2018, "Women", "fruity",
        {"Tangerine": 10, "Grapefruit": 10, "Pineapple": 10},
        {"Coconut": 13, "Jasmine": 14, "Vanilla": 13},
        {"Amber": 10, "Musk": 10, "Sandalwood": 10}),

    # Bath & Body Works (실제 인기 제품들)
    ("Bath & Body Works", "A Thousand Wishes", 2010, "Women", "floral",
        {"Pink Prosecco": 10, "Sparkling Quince": 10, "Crystal Peonies": 10},
        {"Freesia": 13, "Almond Créme": 14, "Sugared Sandalwood": 13},
        {"Vanilla": 10, "Amber": 10, "Musk": 10}),
    ("Bath & Body Works", "Japanese Cherry Blossom", 1999, "Women", "floral",
        {"Cherry Blossom": 10, "Asian Pear": 10, "Fuji Apple": 10},
        {"Jasmine": 13, "Mimosa Petals": 14, "White Flowers": 13},
        {"Sandalwood": 10, "Vanilla": 10, "Musk": 10}),
    ("Bath & Body Works", "Warm Vanilla Sugar", 1999, "Women", "oriental",
        {"Vanilla": 10, "White Orchid": 10, "Sparkling Sugar": 10},
        {"Jasmine": 13, "Caramel": 14, "Creamy Sandalwood": 13},
        {"Vanilla": 10, "Musk": 10, "Amber": 10}),

    # Victoria's Secret
    ("Victoria's Secret", "Bombshell", 2010, "Women", "floral",
        {"Grapefruit": 10, "Pineapple": 10, "Tangerine": 10},
        {"Red Berries": 13, "Jasmine": 14, "Lily of the Valley": 13},
        {"Vanilla Orchid": 10, "Musk": 10, "Woody Accord": 10}),
    ("Victoria's Secret", "Tease", 2010, "Women", "floral",
        {"Black Vanilla": 10, "Pear": 10, "Bamboo": 10},
        {"Gardenia Petals": 13, "Blooming Freesia": 14, "Frozen Pear": 13},
        {"Rich Woods": 10, "Vanilla": 10, "Musk": 10}),
    ("Victoria's Secret", "Love Spell", 2005, "Women", "fruity",
        {"Peach": 10, "Grapefruit": 10, "Cherry Blossom": 10},
        {"Nectarine": 13, "White Jasmine": 14, "Peach": 13},
        {"Musk": 10, "Vanilla": 10, "Amber": 10}),

    # Maison Francis Kurkdjian
    ("Maison Francis Kurkdjian", "Baccarat Rouge 540", 2015, "Unisex", "oriental",
        {"Jasmine": 10, "Saffron": 10, "Bitter Almond": 10},
        {"Cedarwood": 13, "Jasmine": 14, "Ambergris": 13},
        {"Ambergris": 10, "Woody Notes": 10, "Fir Resin": 10}),
    ("Maison Francis Kurkdjian", "Aqua Universalis", 2009, "Unisex", "fresh",
        {"Sicilian Lemon": 10, "Calabrian Bergamot": 10, "Lily of the Valley": 10},
        {"Mock Orange": 13, "Sweet Pea Flower": 14, "Hedione": 13},
        {"Light Woods": 10, "Musk": 10, "Ambergris": 10}),

    # Creed
    ("Creed", "Aventus", 2010, "Men", "fruity",
        {"Pineapple": 10, "Bergamot": 10, "Black Currant": 10},
        {"Birch": 13, "Patchouli": 14, "Jasmine": 13},
        {"Musk": 10, "Oakmoss": 10, "Ambergris": 10}),
    ("Creed", "Silver Mountain Water", 1995, "Unisex", "fresh",
        {"Bergamot": 10, "Mandarin": 10, "Neroli": 10},
        {"Green Tea": 13, "Blackcurrant": 14, "Sandalwood": 13},
        {"Musk": 10, "Petitgrain": 10, "Galbanum": 10}),
    ("Creed", "Green Irish Tweed", 1985, "Men", "fresh",
        {"Lemon Verbena": 10, "Violet Leaf": 10, "Peppermint": 10},
        {"Iris": 13, "Violet": 14, "Violet Leaf": 13},
        {"Sandalwood": 10, "Ambergris": 10, "Mysore Sandalwood": 10}),

    # Byredo
    ("Byredo", "Gypsy Water", 2008, "Unisex", "woody",
        {"Bergamot": 10, "Lemon": 10, "Juniper Berries": 10},
        {"Incense": 13, "Pine Needles": 14, "Orris": 13},
        {"Vanilla": 10, "Sandalwood": 10, "Amber": 10}),
    ("Byredo", "Bal d'Afrique", 2009, "Unisex", "floral",
        {"Lemon": 10, "African Marigold": 10, "Bergamot": 10},
        {"Violet": 13, "Cyclamen": 14, "Neroli": 13},
        {"Vetiver": 10, "Moroccan Cedarwood": 10, "Musk": 10}),

    # Jo Malone
    ("Jo Malone", "Wood Sage & Sea Salt", 2014, "Unisex", "fresh",
        {"Ambrette Seeds": 10, "Sea Salt": 10, "Grapefruit": 10},
        {"Sage": 13, "Seaweed": 14, "Red Algae": 13},
        {"Driftwood": 10, "Ambergris": 10, "White Musk": 10}),
    ("Jo Malone", "English Pear & Freesia", 2010, "Unisex", "fruity",
        {"King William Pear": 10, "Freesia": 10, "Bergamot": 10},
        {"Wild Hyacinth": 13, "Rose": 14, "Jasmine": 13},
        {"Patchouli": 10, "Rhubarb": 10, "Amber": 10}),
    ("Jo Malone", "Lime Basil & Mandarin", 1999, "Unisex", "fresh",
        {"Lime": 10, "Mandarin": 10, "Basil": 10},
        {"Lilac": 13, "Thyme": 14, "Iris": 13},
        {"Vetiver": 10, "Patchouli": 10, "White Musk": 10}),

    # Le Labo
    ("Le Labo", "Santal 33", 2011, "Unisex", "woody",
        {"Cardamom": 10, "Iris": 10, "Violet": 10},
        {"Papyrus": 13, "Ambrox": 14, "Cedar": 13},
        {"Sandalwood": 10, "Leather": 10, "Musk": 10}),
    ("Le Labo", "Rose 31", 2006, "Unisex", "floral",
        {"Rose": 15, "Cumin": 15},
        {"Cistus": 13, "Olibanum": 14, "Cedar": 13},
        {"Guaiac Wood": 10, "Vetiver": 10, "Musk": 10}),
    ("Le Labo", "Another 13", 2010, "Unisex", "woody",
        {"Ambroxan": 10, "Salicylate": 10, "Muscenone": 10},
        {"Iso E Super": 13, "Jasmine": 14, "Moss": 13},
        {"Woody Ambergris": 10, "Musk": 10, "Amber": 10}),

    # Diptyque
    ("Diptyque", "Philosykos", 1996, "Unisex", "woody",
        {"Fig Leaf": 10, "Fig Tree": 10, "Black Pepper": 10},
        {"Green Notes": 13, "Coconut": 14, "Fig Tree Sap": 13},
        {"Fig Tree Wood": 10, "Woody Notes": 10, "Cedar": 10}),
    ("Diptyque", "Tam Dao", 2003, "Unisex", "woody",
        {"Rose": 10, "Cypress": 10, "Myrtle": 10},
        {"Sandalwood": 13, "Cedar": 14, "Rosewood": 13},
        {"Sandalwood": 10, "Amber": 10, "Musk": 10}),
    ("Diptyque", "Do Son", 2005, "Unisex", "floral",
        {"Orange Blossom": 10, "Pink Pepper": 10, "Marine Notes": 10},
        {"Tuberose": 13, "African Orange Flower": 14, "Iris": 13},
        {"Benzoin": 10, "Pink Musk": 10, "Ambrette": 10}),
]

def find_ingredient_match(recipe_name, available_ingredients):
    """원료명 스마트 매칭"""
    if recipe_name in available_ingredients:
        return recipe_name

    # 변형 시도
    variations = [
        recipe_name.replace(" Oil", ""),
        recipe_name.replace(" Absolute", ""),
        recipe_name.replace(" Extract", ""),
        recipe_name.replace(" Essence", ""),
        recipe_name.replace("ian ", ""),  # Calabrian Bergamot -> Bergamot
        recipe_name.split()[-1],  # Last word (e.g., "Bourbon Vanilla" -> "Vanilla")
        " ".join(recipe_name.split()[:-1]) if len(recipe_name.split()) > 1 else recipe_name,  # Remove last word
    ]

    for var in variations:
        if var in available_ingredients:
            return var

    # Fuzzy matching - 부분 매칭
    for available in available_ingredients:
        if recipe_name.lower() in available.lower() or available.lower() in recipe_name.lower():
            return available

    return None

if __name__ == "__main__":
    print("=== 실제 향수 데이터 대량 입력 ===\n")

    conn = sqlite3.connect('fragrance_ai.db')
    cursor = conn.cursor()

    # 원료 목록
    cursor.execute("SELECT name FROM fragrance_notes")
    available_ingredients = {row[0] for row in cursor.fetchall()}
    print(f"사용 가능한 원료: {len(available_ingredients)}개\n")

    # 기존 향수 확인
    cursor.execute("SELECT brand, name FROM perfumes")
    existing = {(row[0], row[1]) for row in cursor.fetchall()}

    added = 0
    skipped = 0
    missing_ingredients = set()

    for brand, name, year, gender, style, top_notes, heart_notes, base_notes in REAL_PERFUMES:
        if (brand, name) in existing:
            skipped += 1
            continue

        try:
            # perfumes 테이블에 추가
            cursor.execute("""
                INSERT INTO perfumes (brand, name, year, gender, style)
                VALUES (?, ?, ?, ?, ?)
            """, (brand, name, year, gender, style))

            perfume_id = cursor.lastrowid

            # 노트 추가
            used_in_perfume = set()

            for position, notes in [('top', top_notes), ('heart', heart_notes), ('base', base_notes)]:
                for ingredient, concentration in notes.items():
                    matched = find_ingredient_match(ingredient, available_ingredients)
                    if matched and (matched, position) not in used_in_perfume:
                        cursor.execute("""
                            INSERT OR IGNORE INTO perfume_notes
                            (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, ?, ?)
                        """, (perfume_id, matched, position, concentration))
                        used_in_perfume.add((matched, position))
                    elif not matched:
                        missing_ingredients.add(ingredient)

            added += 1

            if added % 50 == 0:
                conn.commit()
                print(f"진행: {added}개 추가")

        except Exception as e:
            print(f"✗ {brand} - {name}: {e}")

    conn.commit()
    conn.close()

    print(f"\n=== 완료 ===")
    print(f"✓ {added}개 향수 추가")
    print(f"⊘ {skipped}개 중복")
    print(f"\n매칭되지 않은 원료 ({len(missing_ingredients)}개):")
    for ing in sorted(missing_ingredients)[:20]:
        print(f"  - {ing}")
    if len(missing_ingredients) > 20:
        print(f"  ... 외 {len(missing_ingredients) - 20}개 더")
