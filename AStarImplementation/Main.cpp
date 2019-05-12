#include <algorithm>
#include <memory>
#include <ctime>
#include <utility>
#include <array>
#include <iostream>
#include <optional>
#include <Windows.h>
#include <random>
#undef min

// task 9 oriented graph

template<typename T, typename Name>
struct NamedType
{
	constexpr NamedType()
	{
	}
	constexpr explicit NamedType(T value) :
		_value(value)
	{
	}
	constexpr NamedType(NamedType const&) = default;
	constexpr NamedType(NamedType&&) = default;
	constexpr NamedType& operator=(NamedType const&) = default;
	constexpr NamedType& operator=(NamedType&&) = default;

	constexpr operator T()
	{
		return _value;
	}

	constexpr T& value()
	{
		return _value;
	}
	constexpr T const& value()const
	{
		return _value;
	}

private:
	T _value{};
};

using VertIndex = NamedType<size_t, struct VertIndexType>;
using PathCost = NamedType<float, struct PathCostType>;
using Color = NamedType<int, struct ColorType>;
using uint = unsigned int;

enum class Tile
{
	Empty = 0,
	Full = 1,
};

template<typename T>
class LinkedList
{
	struct Element
	{
		T data;
		Element* next;
		Element* prev;
	};
public:
	class LinkedIter
	{
	public:
		LinkedIter(Element* node) :
			_currentNode(node)
		{
		}

		LinkedIter operator++()
		{
			_currentNode = _currentNode->next;
			return *this;
		}

		LinkedIter operator--()
		{
			_currentNode = _currentNode->prev;
			return *this;
		}

		bool operator==(LinkedIter const& other)const
		{
			return _currentNode == other._currentNode;
		}

		bool operator!=(LinkedIter const& other)const
		{
			return !(*this == other);
		}

		T& operator*()
		{
			return _currentNode->data;
		}
		T const& operator*()const
		{
			return _currentNode->data;
		}

	private:
		Element* _currentNode = nullptr;
	};

	LinkedList()
	{
	}

	LinkedList(LinkedList const& other)
	{
		Element* curr = &head;
		foreach_node([&](Element* node)->void
		{
			curr.next = new Element{node->data, nullptr, curr};
			curr = curr.next;
		});
	}
	LinkedList(LinkedList&& other)noexcept
	{
		swap(*this, other);
	}
	LinkedList& operator=(LinkedList other)
	{
		swap(*this, other);
		return *this;
	}

	~LinkedList()
	{
		clear();
	}

	friend void swap(LinkedList& left, LinkedList& right)
	{
		using std::swap;
		swap(left.head, right.head);
	}

	void clear()
	{
		Element* curr = head.next;
		do
		{
			Element* t = (curr ? curr->next : nullptr);
			delete curr;
			curr = t;
		}
		while (curr);
		head.next = nullptr;
	}

	void add(T const& elem)
	{
		Element* newElem = new Element{elem, head.next, &head};

		if (head.next)
			head.next->prev = newElem;
		head.next = newElem;
	}

	void add(T&& elem)
	{
		Element* newElem = new Element{std::move(elem), head.next, &head};

		if (head.next)
			head.next->prev = newElem;
		head.next = newElem;
	}

	LinkedIter begin()
	{
		return LinkedIter(head.next);
	}
	LinkedIter end()
	{
		return LinkedIter(nullptr);
	}
	LinkedIter begin()const
	{
		return cbegin();
	}
	LinkedIter end()const
	{
		return cend();
	}
	LinkedIter cbegin()const
	{
		return LinkedIter(head.next);
	}
	LinkedIter cend()const
	{
		return LinkedIter(nullptr);
	}

	size_t size()
	{
		return last().second;
	}

private:
	Element head = {T{}, nullptr, nullptr};

	std::pair<Element const*, size_t> last()const
	{
		size_t count = 0;
		Element const* curr = &head;
		Element const* prev;
		do
		{
			count += 1;
			prev = curr;
			curr = curr->next;
		}
		while (curr);
		return {prev, count - 1};
	}

	template<typename Func>
	void foreach_node(Func&& func)
	{
		Element* curr = &head;
		while (true)
		{
			curr = curr->next;
			if (!curr)
				break;
			func(curr);
		}
	}
};

template<typename T>
class DynArray
{
public:
	DynArray()
	{
	}
	DynArray(size_t size) :
		_size(size),
		_data(new T[size])
	{
	}
	DynArray(DynArray const& other)
	{
		_data = std::make_unique<T[]>(other._size);
		std::copy(other._data.get(), other._data.get() + other._size, _data.get());
		_size = other._size;
	}
	DynArray(DynArray&& other)noexcept
	{
		swap(*this, other);
	}
	DynArray& operator=(DynArray other)
	{
		swap(*this, other);
		return *this;
	}

	// why friend: https://pastebin.com/52Lseck1
	friend void swap(DynArray& left, DynArray& right) noexcept
	{
		using std::swap;
		swap(left._size, right._size);
		swap(left._data, right._data);
	}

	size_t size()const
	{
		return _size;
	}

	T* begin()
	{
		return _data.get();
	}
	T* end()
	{
		return _data.get() + _size;
	}
	T const* begin()const
	{
		return cbegin();
	}
	T const* end()const
	{
		return cend();
	}
	T const* cbegin()const
	{
		return _data.get();
	}
	T const* cend()const
	{
		return _data.get() + _size;
	}

	T const& operator[](size_t i) const
	{
		check(i);
		return _data[i];
	}
	T& operator[](size_t i)
	{
		check(i);
		return _data[i];
	}
private:
	size_t _size = 0;
	std::unique_ptr<T[]> _data = nullptr;

	void check(size_t i)const
	{
#ifdef DEBUG
		if (i >= _size)
			std::terminate();
#endif
	}
};

template<typename T>
class Array2D
{
public:
	Array2D()
	{
	}
	Array2D(size_t rows, size_t cols) :
		_cols(cols),
		_data(rows*cols)
	{
	}

	T& at(size_t row, size_t col)
	{
		return _data[_cols*row + col];
	}

	T const& at(size_t row, size_t col)const
	{
		return _data[_cols*row + col];
	}

	size_t rows()const
	{
		return _data.size() / _cols;
	}

	size_t cols()const
	{
		return _cols;
	}

	void fill(T const& value)
	{
		std::fill(_data.begin(), _data.end(), value);
	}

private:
	DynArray<T> _data;
	size_t _cols = 0;
};

template<typename T>
class List
{
public:
	List()
	{
	}

	void push(T const& toAdd)
	{
		size_t newSize = _size + 1;
		if (newSize > _data.size())
			Expand();

		_data[_size] = toAdd;
		_size = newSize;
	}

	void push(T&& toAdd)
	{
		size_t newSize = _size + 1;
		if (newSize > _data.size())
			Expand();

		_data[_size] = std::move(toAdd);
		_size = newSize;
	}

	T& operator[](size_t i)
	{
		return _data[i];
	}
	T const& operator[](size_t i)const
	{
		return _data[i];
	}

	size_t size()const
	{
		return _size;
	}

	T pop()
	{
		if (size() == 0)
			std::terminate();
		_size -= 1;
		return std::move(_data[_size]);
	}

	void extend(List const& other)
	{
		for (auto const& item : other)
			push(item);
	}

	void extend(List&& other)
	{
		for (auto& item : other)
			push(std::move(item));
	}

	void clear()
	{
		_size = 0;
	}

	std::optional<size_t> index(T const& val)const
	{
		if (auto find = std::find(cbegin(), cend(), val); find != cend())
			return std::distance(cbegin(), find);
		return std::nullopt;
	}

	T* begin()
	{
		return _data.begin();
	}
	T* end()
	{
		return _data.begin() + _size;
	}
	T const* begin()const
	{
		return cbegin();
	}
	T const* end()const
	{
		return cend();
	}
	T const* cbegin()const
	{
		return _data.cbegin();
	}
	T const* cend()const
	{
		return _data.cbegin() + _size;
	}

private:
	DynArray<T> _data;
	size_t _size = 0;

	void Expand()
	{
		size_t new_capacity = _data.size() * 3 / 2 + 1;
		DynArray<T> newArr(new_capacity);
		std::move(_data.cbegin(), _data.cend(), newArr.begin());
		_data = newArr;
	}
};

template<typename T>
class PriorityQueue
{
public:
	PriorityQueue()
	{
	}

	T pop_min()
	{
		swap_elements(_data[0], _data[_data.size() - 1]);
		T e = _data.pop();
		e.queue_index() = std::nullopt;
		move_down(0);
		return e;
	}

	void insert(T const& elem)
	{
		elem.queue_index() = _data.size();
		_data.push(elem);
		move_up(_data.size() - 1);
	}
	void insert(T&& elem)
	{
		elem.queue_index() = _data.size();
		_data.push(std::move(elem));
		move_up(_data.size() - 1);
	}

	// low priority - served first
	void update_priority_decreased(size_t elemInd)
	{
		move_up(elemInd);
	}
	void update_priority_increased(size_t elemInd)
	{
		move_down(elemInd);
	}

	T const& operator[](size_t ind)const
	{
		return _data[ind];
	}

	size_t size()const
	{
		return _data.size();
	}

	void clear()
	{
		_data.clear();
	}

private:
	List<T> _data;

	void check_heap(List<T>const& before)
	{
		bool heap = std::is_heap(_data.cbegin(), _data.cend(), [](T const& x, T const& y)
		{
			return x.priority() > y.priority();
		});
		if (!heap)
		{
			auto printHeap = [](auto const& data)
			{
				size_t level = 1;
				for (size_t i = 0; i < data.size(); i++)
				{
					std::cout.precision(3);
					std::cout << data[i].priority() << " ";

					if (i + 2 == (size_t)std::pow(2, level))
					{
						level += 1;
						std::cout << std::endl;
					}

				}
				std::cout << '\n';
			};
			std::cout << "before:\n";
			printHeap(before);
			std::cout << "after:\n";
			printHeap(_data);
			std::terminate();
		}
	}

	void swap_elements(T& x, T& y)
	{
		using std::swap;
		swap(x.queue_index(), y.queue_index());
		swap(x, y);
	}

	void move_up(size_t node)
	{
		List<T> before = _data;
		while (node > 0 && _data[parent(node)].priority() > _data[node].priority())
		{
			swap_elements(_data[parent(node)], _data[node]);
			node = parent(node);
		}
	}

	void move_down(size_t node)
	{
		while (true)
		{
			size_t smallest = node;
			size_t left = left_child(node), right = right_child(node);
			if (left < _data.size() && _data[left].priority() < _data[node].priority())
				smallest = left_child(node);
			if (right < _data.size() && _data[right].priority() < _data[smallest].priority())
				smallest = right_child(node);

			if (smallest == node)
				break;
			else
				swap_elements(_data[node], _data[smallest]);
			node = smallest;
		}
	}

	static size_t parent(size_t node)
	{
		return (node - 1) / 2;
	}
	static size_t left_child(size_t node)
	{
		return node * 2 + 1;
	}
	static size_t right_child(size_t node)
	{
		return left_child(node) + 1;
	}
};

template<typename K, typename V>
class SuboptimalDictionary
{
	struct Element
	{
		K key;
		V value;

		bool operator==(Element const& other)const
		{
			return key == other.key;
		}
		bool operator!=(Element const& other)const
		{
			return !(*this == other);
		}
	};
public:
	SuboptimalDictionary()
	{
	}

	SuboptimalDictionary(size_t buckets)
	{
		for (size_t i = 0; i < buckets; i++)
			_buckets.push({});
	}

	void insert(K const& key, V const& value)
	{
		if (_buckets.size() == 0 || _values / _buckets.size() > 5)
			expand();

		insert_no_expand(key, value);
	}

	void insert(K&& key, V&& value)
	{
		if (_buckets.size() == 0 || _values / _buckets.size() > 5)
			expand();

		insert_no_expand(std::move(key), std::move(value));
	}

	std::optional<V*> get(K const& key)
	{
		size_t hash = key.hash();
		if (std::optional<size_t> i = _buckets[hash % _buckets.size()].index(Element{key}))
			return &_buckets[hash % _buckets.size()][i.value()].value;
		return std::nullopt;
	}
	std::optional<V const*> get(K const& key)const
	{
		size_t hash = key.hash();
		if (std::optional<size_t> i = _buckets[hash % _buckets.size()].index(Element{key}))
			return &_buckets[hash % _buckets.size()][i.value()].value;
		return std::nullopt;
	}

	void clear()
	{
		for (auto& item : _buckets)
			item.clear();
		_values = 0;
	}

private:
	List<List<Element>> _buckets;
	size_t _values = 0;

	void insert_no_expand(K const& key, V const& value)
	{
		_values += 1;
		_buckets[key.hash() % _buckets.size()].push(Element{key,value});
	}

	void insert_no_expand(K&& key, V&& value)
	{
		_values += 1;
		_buckets[key.hash() % _buckets.size()].push(Element{std::move(key), std::move(value)});
	}

	void expand()
	{
		SuboptimalDictionary newDict(_buckets.size() * 3 / 2 + 1);
		for (auto& bucket : _buckets)
			for (auto& item : bucket)
			{
				newDict.insert_no_expand(std::move(item.key), std::move(item.value));
			}
		*this = newDict;
	}
};

struct Vec2
{
public:
	int x = 0, y = 0;

	constexpr Vec2()
	{
	}
	constexpr Vec2(int x, int y) :
		x(x), y(y)
	{
	}

	constexpr size_t hash()const
	{
		return x ^ (y << 16);
	}

	int distance_min(Vec2 const& other)const
	{
		return std::min(std::abs(x - other.x), std::abs(y - other.y));
	}

	int distance_manhattan(Vec2 const& other)const
	{
		return std::abs(x - other.x) + std::abs(y - other.y);
	}

	float distance_euclidean(Vec2 const& other)const
	{
		return std::sqrt((x - other.x)*(x - other.x) + (y - other.y)*(y - other.y));
	}

	constexpr friend Vec2 operator+(Vec2 const& l, Vec2 const& r)
	{
		return Vec2(l.x + r.x, l.y + r.y);
	}
	constexpr friend Vec2 operator-(Vec2 const& l, Vec2 const& r)
	{
		return Vec2(l.x - r.x, l.y - r.y);
	}

	constexpr friend bool operator==(Vec2 const& l, Vec2 const& r)
	{
		return l.x == r.x && l.y == r.y;
	}
	constexpr friend bool operator!=(Vec2 const& l, Vec2 const& r)
	{
		return !(l == r);
	}

	friend std::ostream& operator<<(std::ostream& str, Vec2 const& vec)
	{
		str << vec.x << ' ' << vec.y;
		return str;
	}
};

float distance(Vec2 x, Vec2 y)
{
	return x.distance_euclidean(y);
}

class GraphElement
{
public:
	virtual void print(std::ostream& str)const = 0;
};

class GraphVertex :
	public GraphElement
{
public:
	GraphVertex()
	{
	}

	GraphVertex(Vec2 pos) :
		_pos(pos)
	{
	}

	GraphVertex(GraphVertex const&) = default;

	// Inherited via GraphElement
	virtual void print(std::ostream & str) const override
	{
		str << "Vertex. Pos: " << _pos << "\n";
	}

private:
	Vec2 _pos;
};

class GraphEdge :
	public GraphElement
{
public:
	GraphEdge()
	{
	}

	GraphEdge(Vec2 from, Vec2 to) :
		_from(from),
		_to(to)
	{
	}

	GraphEdge(GraphEdge const&) = default;

	// Inherited via GraphElement
	virtual void print(std::ostream & str) const override
	{
		str << "Edge. From: " << _from << "; To: " << _to << '\n';
	}

private:
	Vec2 _from, _to;
};

class StaticGraph
{
	struct GraphVert
	{
		Vec2 pos;
		List<std::pair<VertIndex, PathCost>> connected;
	};

	class Vertex
	{
	public:
		Vertex()
		{
		}

		Vertex(Vec2 pos, VertIndex from, PathCost gscore) :
			_pos(pos),
			_from(from),
			_gscore(gscore)
		{
		}
		Vertex(Vec2 pos, PathCost gscore) :
			_pos(pos),
			_gscore(gscore)
		{
		}

		Vec2 pos()const
		{
			return _pos;
		}

		PathCost gscore()const
		{
			return _gscore;
		}

		std::optional<VertIndex> from()const
		{
			return _from;
		}

		void set_from(VertIndex from, PathCost newGscore)
		{
			_from = from;
			_gscore = newGscore;
		}

		bool is_explored()const
		{
			return _in_closed;
		}

		void explore()
		{
			_in_closed = true;
		}

		std::optional<size_t>const& queue_index()const
		{
			return _queueIndex;
		}
		std::optional<size_t>& queue_index()
		{
			return _queueIndex;
		}

	private:
		PathCost _gscore = PathCost(0);
		Vec2 _pos;
		std::optional<VertIndex> _from;
		bool _in_closed = false;
		std::optional<size_t> _queueIndex;
	};

	struct VertWrapper
	{
		VertIndex ind;
		List<Vertex>* allNodes = nullptr;
		PathCost hscore;

		VertWrapper()
		{
		}
		VertWrapper(List<Vertex>& nodes, VertIndex ind, PathCost hscore) :
			allNodes(&nodes),
			ind(ind),
			hscore(hscore)
		{
		}
		float priority()const
		{
			return (*allNodes)[ind.value()].gscore().value() + hscore.value();
		}
		std::optional<size_t>& queue_index()
		{
			return (*allNodes)[ind.value()].queue_index();
		}
		std::optional<size_t>const& queue_index()const
		{
			return (*allNodes)[ind.value()].queue_index();
		}
	};

public:
	StaticGraph()
	{
	}

	void print_graph_elements(std::ostream& str)const
	{
		for (auto const& item : _graphElements)
			item->print(str);
	}

	void add_vertex(Vec2 pos, List<std::pair<VertIndex, PathCost>>const& connected)
	{
		_adjList.push({pos,connected});
		add_graph_element(std::make_unique<GraphVertex>(pos));
	}

	void finish()
	{
		for (auto const& vertex : _adjList)
			for (auto const& edge : vertex.connected)
				add_graph_element(std::make_unique<GraphEdge>(vertex.pos, _adjList[edge.first.value()].pos));
	}

	void clear()
	{
		_adjList.clear();
		_graphElements.clear();
	}

	std::optional<VertIndex> get_index(Vec2 pos)const
	{
		if (auto find = std::find_if(_adjList.cbegin(), _adjList.cend(), [&](GraphVert const& item)->bool
		{
			return item.pos == pos;
		}); find != _adjList.cend())
			return VertIndex(std::distance(_adjList.cbegin(), find));
		else
			return std::nullopt;
	}

	template<typename Func>
	void foreach_adjacent(VertIndex index, Func&& func)const
	{
		for (auto&[vert, path] : _adjList[index.value()].connected)
			func(vert, path);
	}

	std::optional<std::pair<List<Vec2>, List<Vec2>>> find_path(VertIndex from, VertIndex to, bool useInferiorDijkstra)const
	{
		// cached containers
		thread_local List<Vertex> allNodes;
		thread_local SuboptimalDictionary<Vec2, VertIndex> posToNode;
		thread_local PriorityQueue<VertWrapper> openSet;
		allNodes.clear();
		posToNode.clear();
		openSet.clear();

		allNodes.push(Vertex(_adjList[from.value()].pos, PathCost(0)));
		posToNode.insert(_adjList[from.value()].pos, VertIndex(0));

		openSet.insert(VertWrapper(allNodes, VertIndex(0), PathCost(distance(_adjList[from.value()].pos, _adjList[to.value()].pos))));
		bool success = false;
		VertIndex current;
		while (openSet.size() > 0)
		{
			current = openSet.pop_min().ind;

			allNodes[current.value()].explore();

			if (allNodes[current.value()].pos() == _adjList[to.value()].pos)
			{
				success = true;
				break;
			}

			auto currentInAdjList = get_index(allNodes[current.value()].pos());
			if (!currentInAdjList)
				std::terminate(); // shouldn't happen

			foreach_adjacent(*currentInAdjList, [&](VertIndex neighbInd, PathCost costToNeighb)->void
			{
				float gscore = allNodes[current.value()].gscore().value() + costToNeighb.value();
				float hscore = useInferiorDijkstra ? 0 : distance(_adjList[neighbInd.value()].pos, _adjList[to.value()].pos);

				VertIndex neighb;
				if (auto n = posToNode.get(_adjList[neighbInd.value()].pos); !n)
				{
					// new node discovered
					allNodes.push(Vertex(_adjList[neighbInd.value()].pos, current, PathCost(gscore)));
					posToNode.insert(_adjList[neighbInd.value()].pos, VertIndex(allNodes.size() - 1));
					neighb = VertIndex(allNodes.size() - 1);

					openSet.insert(VertWrapper(allNodes, neighb, PathCost(hscore)));

					allNodes[neighb.value()].set_from(current, PathCost(gscore));
				}
				else
				{
					neighb = *n.value();

					if (allNodes[neighb.value()].is_explored()				// if in the closed set
						||													// or
						gscore >= allNodes[neighb.value()].gscore().value())// this path is worse than previous
						return;

					allNodes[neighb.value()].set_from(current, PathCost(gscore));
					openSet.update_priority_decreased(allNodes[neighb.value()].queue_index().value());
				}
			});
		}

		if (success)
		{
			List<Vec2> path;

			std::optional<VertIndex> trace = current;
			while (trace)
			{
				path.push(allNodes[trace.value().value()].pos());
				trace = allNodes[trace.value().value()].from();
			}

			List<Vec2> consideredNodes;
			for (size_t i = 0; i < allNodes.size(); i++)
				consideredNodes.push(allNodes[i].pos());

			return std::pair(path, consideredNodes);
		}
		else
		{
			return std::nullopt;
		}
	}

	size_t size()const
	{
		return _adjList.size();
	}

	Array2D<float> adj_matrix()const
	{
		Array2D<float> matrix(_adjList.size(), _adjList.size());
		matrix.fill(0);
		for (size_t i = 0; i < _adjList.size(); i++)
		{
			auto const& conn = _adjList[i].connected;
			for (auto c : conn)
				matrix.at(i, c.first.value()) = c.second.value();
		}
		return matrix;
	}

	Array2D<int> inc_matrix()const
	{
		// pair<from, to>
		List<std::pair<VertIndex, VertIndex>> allConnections;
		for (size_t i = 0; i < _adjList.size(); i++) // should be "for(auto&[i, conn] : std::enumerate(_adjList))" (C++20 ranges?)
			for (auto const& conn : _adjList[i].connected)
				allConnections.push({VertIndex(i), conn.first});

		Array2D<int> matrix(_adjList.size(), allConnections.size());
		matrix.fill(0);
		for (size_t i = 0; i < allConnections.size(); i++) // should be "for(auto&[i, conn] : std::enumerate(_adjList))" (C++20 ranges?)
		{
			matrix.at(allConnections[i].first, i) = -1;
			matrix.at(allConnections[i].second, i) = 1;
		}
		return matrix;
	}

private:
	List<GraphVert> _adjList;
	LinkedList<std::unique_ptr<GraphElement>> _graphElements;

	void add_graph_element(std::unique_ptr<GraphElement> element)
	{
		_graphElements.add(std::move(element));
	}
};

class DisplayBuffer
{
public:
	DisplayBuffer(size_t rows, size_t columns) :
		_characters(rows, columns)
	{
	}

	void set(size_t row, size_t col, Color color, char ch)
	{
		_characters.at(row, col) = {color,ch};
	}

	void print(HANDLE handle)const
	{
		for (size_t row = 0; row < _characters.rows(); row++)
		{
			for (size_t col = 0; col < _characters.cols(); col++)
			{
				auto at = _characters.at(row, col);
				SetConsoleTextAttribute(handle, at.first.value());
				std::cout << at.second;
			}
			SetConsoleTextAttribute(handle, 7);
			std::cout << '\n';
		}
	}

private:
	Array2D<std::pair<Color, char>> _characters;
};

class Board
{
	static constexpr std::array<Vec2, 8> offsetsAll{Vec2{-1,-1},{-1,1},{-1,0},{1,1},{1,0},{1,-1},{0,1},{0,-1}};
	static constexpr std::array<Vec2, 4> offsetsSquare{Vec2{-1,0},{1,0},{0,1},{0,-1}};

public:
	Board(size_t rows, size_t cols, std::default_random_engine& random, size_t smoothsteps = 3) :
		_data(rows, cols)
	{
		generate(smoothsteps, random);
	}

	void draw(DisplayBuffer& buff)const
	{
		for (size_t row = 0; row < _data.rows(); row++)
			for (size_t col = 0; col < _data.cols(); col++)
			{
				buff.set(row, col, Color(7), _data.at(row, col) == Tile::Empty ? '~' : '#');
			}
	}

	std::optional<std::pair<List<Vec2>, List<Vec2>>> find_path(Vec2 from, Vec2 to, bool useInferiorDijkstra = false) const
	{
		if (auto i1 = _graph.get_index(from))
			if (auto i2 = _graph.get_index(to))
				return _graph.find_path(i1.value(), i2.value(), useInferiorDijkstra);
		return std::nullopt;
	}

	Vec2 get_random_empty_tile(std::default_random_engine& random)const
	{
		std::uniform_int_distribution distrX(size_t(0), _data.cols());
		std::uniform_int_distribution distrY(size_t(0), _data.rows());

		size_t x;
		size_t y;
		do
		{
			x = distrX(random);
			y = distrY(random);
		}
		while (_data.at(y, x) != Tile::Empty);
		return Vec2(x, y);
	}

	StaticGraph const& graph()const
	{
		return _graph;
	}

private:
	Array2D<Tile> _data;
	StaticGraph _graph;

	template<typename Func, size_t Sz>
	void foreach_empty_neighbour(Vec2 pos, std::array<Vec2, Sz>const& offsets, Func&& func)const
	{
		for (auto& item : offsets)
		{
			Vec2 neighbPos = pos + item;
			if (neighbPos.x >= 0 && neighbPos.y >= 0
				&&
				neighbPos.y < _data.rows() && neighbPos.x < _data.cols())
				if (_data.at(neighbPos.y, neighbPos.x) == Tile::Empty)
					func(neighbPos);
		}
	}

	void generate(size_t smoothsteps, std::default_random_engine& random)
	{
		std::uniform_int_distribution distr(0, 1);
		for (size_t row = 0; row < _data.rows(); row++)
			for (size_t col = 0; col < _data.cols(); col++)
				_data.at(row, col) = (distr(random) ? Tile::Empty : Tile::Full);

		Array2D<Tile> new_tiles;
		for (size_t k = 0; k < smoothsteps; k++)
		{
			new_tiles = _data;
			for (size_t row = 0; row < _data.rows(); row++)
				for (size_t col = 0; col < _data.cols(); col++)
				{
					size_t neighb_count = 0;
					for (size_t i = 0; i < offsetsAll.size(); i++)
					{
						auto offset = offsetsAll[i];

						size_t nextrow = offset.y + row,
							nextcol = offset.x + col;

						if (nextrow < _data.rows() && nextcol < _data.cols())
							if (_data.at(nextrow, nextcol) == Tile::Full)
								neighb_count += 1;
					}
					new_tiles.at(row, col) = (neighb_count > 4 ? Tile::Full :
											  Tile::Empty);

				}
			_data = new_tiles;
		}

		update_graph();
	}

	void update_graph()
	{
		_graph.clear();
		for (size_t row = 0; row < _data.rows(); row++)
			for (size_t col = 0; col < _data.cols(); col++)
			{
				Vec2 curr(col, row);
				List<std::pair<VertIndex, PathCost>> connected;
				foreach_empty_neighbour(curr, offsetsAll, [&](Vec2 neighb)
				{
					connected.push({VertIndex(neighb.y * _data.cols() + neighb.x), PathCost(distance(curr, neighb))});
				});
				_graph.add_vertex(curr, connected);
			}
		_graph.finish();
	}
};

void clear_screen(char fill = ' ')
{
	COORD tl = {0,0};
	CONSOLE_SCREEN_BUFFER_INFO s;
	HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(console, &s);
	DWORD written, cells = s.dwSize.X * s.dwSize.Y;
	FillConsoleOutputCharacter(console, fill, cells, tl, &written);
	FillConsoleOutputAttribute(console, s.wAttributes, cells, tl, &written);
	SetConsoleCursorPosition(console, tl);
}

void draw_points(List<Vec2>const& path, DisplayBuffer& buff, Color color)
{
	for (size_t i = 0; i < path.size(); i++)
	{
		buff.set(path[i].y, path[i].x, color, '~');
	}
}

void draw_points(List<Vec2>const& path, DisplayBuffer& buff, Color color, Color colorStart, Color colorEnd)
{
	for (size_t i = 1; i < path.size() - 1; i++)
	{
		buff.set(path[i].y, path[i].x, color, '~');
	}
	buff.set(path[0].y, path[0].x, colorEnd, 'f');
	buff.set(path[path.size() - 1].y, path[path.size() - 1].x, colorStart, 's');
}

template<typename T>
void draw_matrix(Array2D<T>const& matr, size_t width = 0)
{
	for (size_t row = 0; row < matr.rows(); row++)
	{
		for (size_t col = 0; col < matr.cols(); col++)
		{
			std::cout.precision(2);
			std::cout.width(width);
			std::cout << matr.at(row, col) << ' ';
		}
		std::cout << '\n';
	}
}

int main()
{
	constexpr uint defaultGridSize = 50;
	std::default_random_engine random(std::random_device{}());

	// input
	uint gridSize;
	std::cout << "Enter grid size: \n";
	std::cin >> gridSize;
	if (std::cin.fail())
	{
		gridSize = defaultGridSize;
		std::cout << "Some error occured, using default grid size\n";
		std::cin.clear();
	}
	// -input

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	DisplayBuffer buffer(gridSize, gridSize);

	char userInput = 'r';
	Board board(gridSize, gridSize, random);
	List<Vec2> path;
	Vec2 point1 = board.get_random_empty_tile(random), point2 = board.get_random_empty_tile(random);

	auto drawPath = [&](bool isDij)->void
	{
		if (auto pair = board.find_path(point1, point2, isDij))
		{
			auto&[path, considered] = *pair;

			board.draw(buffer);
			draw_points(considered, buffer, Color(160));
			draw_points(path, buffer, Color(200), Color(206), Color(207));
		}
	};
	do
	{
		clear_screen();
		switch (userInput)
		{
			case 'r':
				board = Board(gridSize, gridSize, random);
				board.draw(buffer);

				point1 = board.get_random_empty_tile(random);
				point2 = board.get_random_empty_tile(random);
				break;

			case 'd':
				draw_matrix(board.graph().adj_matrix(), 3);
				break;

			case 'i':
				draw_matrix(board.graph().inc_matrix(), 2);
				break;

			case 'a':
				drawPath(false);
				break;

			case 'j':
				drawPath(true);
				break;

			case 'p':
				point1 = board.get_random_empty_tile(random);
				point2 = board.get_random_empty_tile(random);
				break;

			case 't':
				board.graph().print_graph_elements(std::cout);
				break;

			default:
				break;
		}
		buffer.print(hConsole);

		std::cout << "Options: \nq - exit\nr - generate new board\np - pick 2 random points\na - build path using superior A*\nj - build path using inferior Dijkstra\nd - print adjacency matrix for this board\ni - print incidence matrix\nt - print all graph elements polymorphically\n";
		std::cin >> userInput;
	}
	while (userInput != 'q');
}
