// src/schema.js — E-Commerce GraphQL Schema
// Products, Categories, Cart, Orders, Reviews for the demo store

const { buildSchema } = require('graphql');

const typeDefs = `
  type Product {
    id: ID!
    name: String!
    description: String!
    price: Float!
    originalPrice: Float
    category: Category!
    images: [String!]!
    rating: Float!
    reviewCount: Int!
    inStock: Boolean!
    tags: [String!]!
    reviews: [Review!]!
  }

  type Category {
    id: ID!
    name: String!
    slug: String!
    description: String!
    products: [Product!]!
    productCount: Int!
  }

  type Customer {
    id: ID!
    name: String!
    email: String!
    phone: String
    address: Address
    orders: [Order!]!
    cart: Cart
    role: String!
  }

  type Address {
    street: String!
    city: String!
    state: String!
    zip: String!
    country: String!
  }

  type Cart {
    id: ID!
    customer: Customer!
    items: [CartItem!]!
    total: Float!
    itemCount: Int!
  }

  type CartItem {
    id: ID!
    product: Product!
    quantity: Int!
    subtotal: Float!
  }

  type Order {
    id: ID!
    customer: Customer!
    items: [OrderItem!]!
    total: Float!
    status: String!
    createdAt: String!
    shippingAddress: Address!
    paymentMethod: String!
  }

  type OrderItem {
    id: ID!
    product: Product!
    quantity: Int!
    priceAtPurchase: Float!
    subtotal: Float!
  }

  type Review {
    id: ID!
    product: Product!
    customer: Customer!
    rating: Int!
    title: String!
    body: String!
    helpful: Int!
    createdAt: String!
  }

  type AuthPayload {
    token: String!
    customer: Customer!
  }

  type Query {
    products(categoryId: ID, search: String, limit: Int, offset: Int): [Product!]!
    product(id: ID!): Product
    categories: [Category!]!
    category(slug: String!): Category
    customer(id: ID!): Customer
    customers: [Customer!]!
    order(id: ID!): Order
    orders(customerId: ID): [Order!]!
    cart(customerId: ID!): Cart
    reviews(productId: ID!): [Review!]!
    featuredProducts: [Product!]!
    searchProducts(query: String!): [Product!]!
  }

  type Mutation {
    login(email: String!, password: String!): AuthPayload
    addToCart(customerId: ID!, productId: ID!, quantity: Int!): Cart
    removeFromCart(customerId: ID!, cartItemId: ID!): Cart
    updateCartItem(customerId: ID!, cartItemId: ID!, quantity: Int!): Cart
    placeOrder(customerId: ID!, paymentMethod: String!): Order
    addReview(productId: ID!, customerId: ID!, rating: Int!, title: String!, body: String!): Review
    updateOrderStatus(orderId: ID!, status: String!): Order
  }
`;

// ── Seed Data ────────────────────────────────────────────────────────────────

const categories = [
  { id: '1', name: 'Electronics', slug: 'electronics', description: 'Latest gadgets and tech' },
  { id: '2', name: 'Clothing', slug: 'clothing', description: 'Fashion for everyone' },
  { id: '3', name: 'Books', slug: 'books', description: 'Knowledge at your fingertips' },
  { id: '4', name: 'Home & Living', slug: 'home', description: 'Make your space beautiful' },
];

const products = [
  { id: '1', name: 'iPhone 16 Pro', price: 999.99, originalPrice: 1099.99, categoryId: '1', images: ['iphone16pro.jpg'], rating: 4.8, reviewCount: 234, inStock: true, tags: ['apple', 'smartphone', '5g'], description: 'The most advanced iPhone ever.' },
  { id: '2', name: 'MacBook Pro M4', price: 1999.00, originalPrice: 2199.00, categoryId: '1', images: ['macbook.jpg'], rating: 4.9, reviewCount: 189, inStock: true, tags: ['apple', 'laptop', 'm4'], description: 'Power meets portability.' },
  { id: '3', name: 'Sony WH-1000XM5', price: 349.99, originalPrice: 399.99, categoryId: '1', images: ['sony_headphones.jpg'], rating: 4.7, reviewCount: 521, inStock: true, tags: ['sony', 'headphones', 'anc'], description: 'Industry-leading noise cancellation.' },
  { id: '4', name: 'Samsung 4K OLED', price: 1299.00, originalPrice: 1499.00, categoryId: '1', images: ['tv.jpg'], rating: 4.6, reviewCount: 88, inStock: false, tags: ['samsung', 'tv', '4k'], description: 'Stunning 65-inch OLED display.' },
  { id: '5', name: 'iPad Air M2', price: 599.00, originalPrice: 649.00, categoryId: '1', images: ['ipad.jpg'], rating: 4.7, reviewCount: 312, inStock: true, tags: ['apple', 'tablet', 'm2'], description: 'Thin, light, incredibly capable.' },
  { id: '6', name: 'Classic Denim Jacket', price: 79.99, originalPrice: 99.99, categoryId: '2', images: ['denim.jpg'], rating: 4.4, reviewCount: 67, inStock: true, tags: ['denim', 'jacket', 'classic'], description: 'Timeless style for any occasion.' },
  { id: '7', name: 'Premium White Tee', price: 29.99, originalPrice: 39.99, categoryId: '2', images: ['tee.jpg'], rating: 4.5, reviewCount: 143, inStock: true, tags: ['cotton', 'basics', 'tshirt'], description: 'Ultra-soft 100% organic cotton.' },
  { id: '8', name: 'Slim Chinos', price: 59.99, originalPrice: 79.99, categoryId: '2', images: ['chinos.jpg'], rating: 4.3, reviewCount: 92, inStock: true, tags: ['pants', 'slim', 'chinos'], description: 'Modern slim fit for office or casual.' },
  { id: '9', name: 'Leather Sneakers', price: 119.99, originalPrice: 149.99, categoryId: '2', images: ['sneakers.jpg'], rating: 4.6, reviewCount: 204, inStock: true, tags: ['shoes', 'leather', 'sneakers'], description: 'Handcrafted Italian leather.' },
  { id: '10', name: 'Summer Dress', price: 49.99, originalPrice: 69.99, categoryId: '2', images: ['dress.jpg'], rating: 4.7, reviewCount: 176, inStock: true, tags: ['summer', 'dress', 'floral'], description: 'Light and breezy for warm days.' },
  { id: '11', name: 'Clean Code', price: 34.99, originalPrice: 44.99, categoryId: '3', images: ['cleancode.jpg'], rating: 4.8, reviewCount: 892, inStock: true, tags: ['programming', 'software', 'craft'], 'description': 'A Handbook of Agile Software Craftsmanship.' },
  { id: '12', name: 'Dune', price: 14.99, originalPrice: 19.99, categoryId: '3', images: ['dune.jpg'], rating: 4.9, reviewCount: 2341, inStock: true, tags: ['scifi', 'classic', 'frank-herbert'], description: 'Epic sci-fi masterpiece.' },
  { id: '13', name: 'Atomic Habits', price: 19.99, originalPrice: 26.99, categoryId: '3', images: ['habits.jpg'], rating: 4.8, reviewCount: 5621, inStock: true, tags: ['self-help', 'habits', 'productivity'], description: 'Build good habits, break bad ones.' },
  { id: '14', name: 'The Pragmatic Programmer', price: 39.99, originalPrice: 49.99, categoryId: '3', images: ['pragmatic.jpg'], rating: 4.7, reviewCount: 447, inStock: true, tags: ['programming', 'career', 'software'], description: 'Your journey to mastery.' },
  { id: '15', name: 'Designing Data-Intensive Applications', price: 44.99, originalPrice: 54.99, categoryId: '3', images: ['ddia.jpg'], rating: 4.9, reviewCount: 1203, inStock: true, tags: ['databases', 'systems', 'distributed'], description: 'The definitive guide to scalable systems.' },
  { id: '16', name: 'Minimalist Desk Lamp', price: 89.99, originalPrice: 109.99, categoryId: '4', images: ['lamp.jpg'], rating: 4.5, reviewCount: 234, inStock: true, tags: ['lamp', 'desk', 'minimalist'], description: 'Clean lines, warm light.' },
  { id: '17', name: 'Bamboo Plant Stand', price: 49.99, originalPrice: 59.99, categoryId: '4', images: ['stand.jpg'], rating: 4.6, reviewCount: 89, inStock: true, tags: ['plant', 'bamboo', 'home'], description: 'Sustainable and stylish.' },
  { id: '18', name: 'Ceramic Coffee Set', price: 69.99, originalPrice: 89.99, categoryId: '4', images: ['coffee.jpg'], rating: 4.7, reviewCount: 156, inStock: true, tags: ['ceramic', 'coffee', 'kitchen'], description: 'Handmade artisan ceramic set.' },
  { id: '19', name: 'Smart Air Purifier', price: 199.99, originalPrice: 249.99, categoryId: '4', images: ['purifier.jpg'], rating: 4.8, reviewCount: 78, inStock: true, tags: ['air', 'smart', 'health'], description: 'Breathe cleaner, sleep better.' },
  { id: '20', name: 'Throw Blanket', price: 39.99, originalPrice: 54.99, categoryId: '4', images: ['blanket.jpg'], rating: 4.9, reviewCount: 312, inStock: true, tags: ['blanket', 'cozy', 'home'], description: 'Incredibly soft merino wool.' },
];

const customers = [
  { id: '1', name: 'Alice Chen', email: 'alice@demo.com', phone: '555-0101', role: 'customer', password: 'pass123', address: { street: '123 Tech Ave', city: 'San Francisco', state: 'CA', zip: '94102', country: 'US' } },
  { id: '2', name: 'Bob Martinez', email: 'bob@demo.com', phone: '555-0102', role: 'customer', password: 'pass123', address: { street: '456 Market St', city: 'New York', state: 'NY', zip: '10001', country: 'US' } },
  { id: '3', name: 'Carol Johnson', email: 'carol@demo.com', phone: '555-0103', role: 'customer', password: 'pass123', address: { street: '789 Oak Rd', city: 'Chicago', state: 'IL', zip: '60601', country: 'US' } },
  { id: '4', name: 'Admin User', email: 'admin@demo.com', phone: '555-0000', role: 'admin', password: 'admin123', address: { street: '1 Admin Blvd', city: 'Austin', state: 'TX', zip: '73301', country: 'US' } },
];

const reviews = [
  { id: '1', productId: '1', customerId: '2', rating: 5, title: 'Amazing phone!', body: 'The camera quality is absolutely stunning. Best upgrade I have made.', helpful: 45, createdAt: '2024-01-15' },
  { id: '2', productId: '1', customerId: '3', rating: 4, title: 'Great but pricey', body: 'Performance is top-notch but wish it was a bit cheaper.', helpful: 23, createdAt: '2024-01-20' },
  { id: '3', productId: '2', customerId: '1', rating: 5, title: 'M4 is a beast', body: 'Handles everything I throw at it. Battery life is incredible.', helpful: 67, createdAt: '2024-01-18' },
  { id: '4', productId: '11', customerId: '1', rating: 5, title: 'Must-read for devs', body: 'Changed how I write code entirely. Essential reading.', helpful: 89, createdAt: '2024-01-10' },
  { id: '5', productId: '13', customerId: '2', rating: 5, title: 'Life changing', body: 'Simple frameworks that actually work. Highly recommend.', helpful: 134, createdAt: '2024-01-12' },
];

const orders = [
  { id: '1', customerId: '1', status: 'delivered', createdAt: '2024-01-05', paymentMethod: 'card', items: [{ id: '1', productId: '1', quantity: 1, priceAtPurchase: 999.99 }, { id: '2', productId: '3', quantity: 1, priceAtPurchase: 349.99 }] },
  { id: '2', customerId: '2', status: 'shipped', createdAt: '2024-01-15', paymentMethod: 'paypal', items: [{ id: '3', productId: '11', quantity: 2, priceAtPurchase: 34.99 }] },
  { id: '3', customerId: '3', status: 'pending', createdAt: '2024-01-22', paymentMethod: 'card', items: [{ id: '4', productId: '16', quantity: 1, priceAtPurchase: 89.99 }] },
];

const carts = {
  '1': { id: 'c1', customerId: '1', items: [{ id: 'ci1', productId: '2', quantity: 1 }] },
  '2': { id: 'c2', customerId: '2', items: [] },
  '3': { id: 'c3', customerId: '3', items: [{ id: 'ci2', productId: '20', quantity: 2 }] },
};

// ── Helpers ──────────────────────────────────────────────────────────────────

const findProduct = id => products.find(p => p.id === id);
const findCategory = id => categories.find(c => c.id === id);
const findCustomer = id => customers.find(c => c.id === id);

function resolveProduct(p) {
  return {
    ...p,
    category: () => findCategory(p.categoryId),
    reviews: () => reviews.filter(r => r.productId === p.id).map(resolveReview),
  };
}

function resolveReview(r) {
  return { ...r, product: () => resolveProduct(findProduct(r.productId)), customer: () => findCustomer(r.customerId) };
}

function resolveOrder(o) {
  const total = o.items.reduce((s, i) => s + i.priceAtPurchase * i.quantity, 0);
  const cust = findCustomer(o.customerId);
  return {
    ...o,
    total,
    customer: () => cust,
    shippingAddress: () => cust?.address || {},
    items: o.items.map(i => ({
      ...i,
      subtotal: i.priceAtPurchase * i.quantity,
      product: () => resolveProduct(findProduct(i.productId)),
    })),
  };
}

function resolveCart(cartData) {
  if (!cartData) return null;
  const customer = findCustomer(cartData.customerId);
  const items = cartData.items.map(ci => {
    const prod = findProduct(ci.productId);
    return { ...ci, subtotal: (prod?.price || 0) * ci.quantity, product: () => resolveProduct(prod) };
  });
  const total = items.reduce((s, i) => s + i.subtotal, 0);
  return { ...cartData, customer: () => customer, items, total, itemCount: items.reduce((s, i) => s + i.quantity, 0) };
}

// ── Resolvers ─────────────────────────────────────────────────────────────────

const resolvers = {
  Query: {
    products: (_, { categoryId, search, limit = 20, offset = 0 }) => {
      let result = products;
      if (categoryId) result = result.filter(p => p.categoryId === categoryId);
      if (search) result = result.filter(p => p.name.toLowerCase().includes(search.toLowerCase()) || p.tags.some(t => t.includes(search.toLowerCase())));
      return result.slice(offset, offset + limit).map(resolveProduct);
    },
    product: (_, { id }) => { const p = findProduct(id); return p ? resolveProduct(p) : null; },
    categories: () => categories.map(c => ({ ...c, products: () => products.filter(p => p.categoryId === c.id).map(resolveProduct), productCount: products.filter(p => p.categoryId === c.id).length })),
    category: (_, { slug }) => { const c = categories.find(x => x.slug === slug); return c ? { ...c, products: () => products.filter(p => p.categoryId === c.id).map(resolveProduct), productCount: products.filter(p => p.categoryId === c.id).length } : null; },
    customer: (_, { id }) => findCustomer(id),
    customers: () => customers,
    order: (_, { id }) => { const o = orders.find(x => x.id === id); return o ? resolveOrder(o) : null; },
    orders: (_, { customerId }) => (customerId ? orders.filter(o => o.customerId === customerId) : orders).map(resolveOrder),
    cart: (_, { customerId }) => resolveCart(carts[customerId]),
    reviews: (_, { productId }) => reviews.filter(r => r.productId === productId).map(resolveReview),
    featuredProducts: () => products.filter(p => p.rating >= 4.7).map(resolveProduct),
    searchProducts: (_, { query }) => products.filter(p => p.name.toLowerCase().includes(query.toLowerCase()) || p.description.toLowerCase().includes(query.toLowerCase()) || p.tags.some(t => t.includes(query.toLowerCase()))).map(resolveProduct),
  },
  Mutation: {
    login: (_, { email, password }) => {
      const cust = customers.find(c => c.email === email && c.password === password);
      if (!cust) throw new Error('Invalid credentials');
      return { token: `mock-jwt-${cust.id}-${Date.now()}`, customer: cust };
    },
    addToCart: (_, { customerId, productId, quantity }) => {
      if (!carts[customerId]) carts[customerId] = { id: `c${customerId}`, customerId, items: [] };
      const cart = carts[customerId];
      const existing = cart.items.find(i => i.productId === productId);
      if (existing) existing.quantity += quantity;
      else cart.items.push({ id: `ci${Date.now()}`, productId, quantity });
      return resolveCart(cart);
    },
    removeFromCart: (_, { customerId, cartItemId }) => {
      if (carts[customerId]) carts[customerId].items = carts[customerId].items.filter(i => i.id !== cartItemId);
      return resolveCart(carts[customerId]);
    },
    updateCartItem: (_, { customerId, cartItemId, quantity }) => {
      const item = carts[customerId]?.items.find(i => i.id === cartItemId);
      if (item) item.quantity = quantity;
      return resolveCart(carts[customerId]);
    },
    placeOrder: (_, { customerId, paymentMethod }) => {
      const cart = carts[customerId];
      if (!cart || !cart.items.length) throw new Error('Cart is empty');
      const orderItems = cart.items.map(i => ({ id: `oi${Date.now()}-${i.id}`, productId: i.productId, quantity: i.quantity, priceAtPurchase: findProduct(i.productId)?.price || 0 }));
      const newOrder = { id: `${orders.length + 1}`, customerId, status: 'pending', createdAt: new Date().toISOString(), paymentMethod, items: orderItems };
      orders.push(newOrder);
      carts[customerId].items = [];
      return resolveOrder(newOrder);
    },
    addReview: (_, { productId, customerId, rating, title, body }) => {
      const review = { id: `${reviews.length + 1}`, productId, customerId, rating, title, body, helpful: 0, createdAt: new Date().toISOString() };
      reviews.push(review);
      return resolveReview(review);
    },
    updateOrderStatus: (_, { orderId, status }) => {
      const order = orders.find(o => o.id === orderId);
      if (!order) throw new Error('Order not found');
      order.status = status;
      return resolveOrder(order);
    },
  },
};

module.exports = { typeDefs, resolvers };
