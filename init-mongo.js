// init-mongo.js
db = db.getSiblingDB('interior_analysis');

// Kolekcja mieszkań
db.apartments.insertMany([
  {
    _id: "apt_001",
    title: "Nowoczesne mieszkanie w centrum",
    description: "Przestronne mieszkanie z widokiem na miasto",
    address: "Warszawa, Śródmieście",
    size: 65,
    rooms: 3,
    created_at: new Date(),
    updated_at: new Date(),
    status: "active"
  },
  {
    _id: "apt_002",
    title: "Kawalerka w stylu skandynawskim",
    description: "Przytulna kawalerka w centrum",
    address: "Kraków, Stare Miasto",
    size: 32,
    rooms: 1,
    created_at: new Date(),
    updated_at: new Date(),
    status: "active"
  }
]);

// Kolekcja zdjęć
db.images.insertMany([
  // Mieszkanie 1
  {
    apartment_id: "apt_001",
    url: "https://example.com/images/apt1_room1.jpg",
    sequence: 1,
    room_type: "unknown",
    style: "unknown",
    analysis_status: "pending",
    created_at: new Date()
  },
  {
    apartment_id: "apt_001",
    url: "https://example.com/images/apt1_room2.jpg",
    sequence: 2,
    room_type: "unknown",
    style: "unknown",
    analysis_status: "pending",
    created_at: new Date()
  },
  // Mieszkanie 2
  {
    apartment_id: "apt_002",
    url: "https://example.com/images/apt2_room1.jpg",
    sequence: 1,
    room_type: "unknown",
    style: "unknown",
    analysis_status: "pending",
    created_at: new Date()
  }
]);

// Kolekcja wyników analizy
db.analysis_results.createIndex({ "apartment_id": 1 }, { unique: true });

print("Database initialized successfully!");